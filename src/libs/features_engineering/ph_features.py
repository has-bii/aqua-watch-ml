import pandas as pd
from typing import Optional, Tuple, List
from scipy import stats
import numpy as np
import logging

logger = logging.getLogger(__name__)

class FeatureEngineeringPH:
    LAG_PERIODS = [1, 2, 3]
    
    WATER_CHANGE_WINDOW_MINUTES = 60
    DEFAULT_IS_POST_WATER_CHANGE_60MIN = False
    DEFAULT_IS_POST_AFTER_FEED_60MIN = False

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
                historical_data, water_change_data, features,
            )

            # Prepare feed features with enhanced debugging
            historical_data = self.prepare_feed_features(historical_data, feed_data, features)

            # Prepare outlier detection
            historical_data = self.detect_outliers(historical_data, 'ph', threshold=2.0)

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
                                           historical_data: pd.DataFrame, 
                                           water_change_df: pd.DataFrame,
                                           features: Optional[List[str]] = None,
                                           index_name: Optional[str] = 'created_at'
                                           ) -> pd.DataFrame:
        """Create features related to water changes with support for multiple events."""
        try:
            if features is not None:
                features.append('is_post_water_change_60min')
                features.append('diff_water_temp_after_change')

            if water_change_df.empty:
                # If no water changes, initialize with default values
                historical_data['is_post_water_change_60min'] = self.DEFAULT_IS_POST_WATER_CHANGE_60MIN
                historical_data['diff_water_temp_after_change'] = 0

                return historical_data

            is_reset_index = False

            # Ensure we have created_at as a column for processing
            if isinstance(historical_data.index, pd.DatetimeIndex):
                historical_data.reset_index(inplace=True)
                historical_data[index_name] = pd.to_datetime(historical_data[index_name], format='ISO8601')
                is_reset_index = True
            
            # Ensure datetime columns are properly formatted
            historical_data[index_name] = pd.to_datetime(historical_data[index_name], format='ISO8601')
            water_change_df = water_change_df.reset_index() if isinstance(water_change_df.index, pd.DatetimeIndex) else water_change_df.copy()
            water_change_df['changed_at'] = pd.to_datetime(water_change_df['changed_at'], format='ISO8601')

            # Initialize features with default values
            historical_data['is_post_water_change_60min'] = self.DEFAULT_IS_POST_WATER_CHANGE_60MIN
            historical_data['diff_water_temp_after_change'] = 0.0

            # Process each water change event
            for _, water_change in water_change_df.iterrows():
                water_change_time = water_change['changed_at']
                water_temp_added = water_change['water_temperature_added']
                
                # Calculate time differences for all records (vectorized)
                time_diffs = (historical_data[index_name] - water_change_time).dt.total_seconds() / 60
                
                # Find records within the time window after this water change
                within_window = (time_diffs >= 0) & (time_diffs <= self.WATER_CHANGE_WINDOW_MINUTES)
                
                # Update features for records within the window
                # Only update if this water change is more recent than previously recorded ones
                closer_to_change = time_diffs < 60  # Within 60 minutes
                update_mask = within_window & closer_to_change
                
                if update_mask.any():
                    historical_data.loc[update_mask, 'is_post_water_change_60min'] = True
                    historical_data.loc[update_mask, 'diff_water_temp_after_change'] = (
                        historical_data.loc[update_mask, 'water_temperature'] - water_temp_added
                    )

            # Restore the original index structure
            if is_reset_index:
                historical_data.set_index(index_name, inplace=True)

            return historical_data
        except Exception as e:
            raise ValueError(f"Error preparing features with water change data: {e}")

    def prepare_feed_features(self,
                          historical_data: pd.DataFrame,
                          feed_data: pd.DataFrame,
                          features: Optional[List[str]]) -> pd.DataFrame:
        """
        Prepare features related to feed data with boolean indicator for 60-minute window.
        
        Args:
            historical_data: DataFrame with pH measurements and timestamps
            feed_data: DataFrame with feed events and 'fed_at' timestamps
            
        Returns:
            DataFrame with added 'is_post_after_feed_60min' boolean feature
        """
        try:
            # Work with copies to avoid modifying original data
            df_working = historical_data.copy()
            feed_df_working = feed_data.copy()

            if features is not None:
                features.append('is_post_after_feed_60min')

            feed_was_indexed = isinstance(feed_df_working.index, pd.DatetimeIndex)
            if feed_was_indexed:
                feed_df_working.reset_index(inplace=True)

            if feed_df_working.empty:
                historical_data['is_post_after_feed_60min'] = self.DEFAULT_IS_POST_AFTER_FEED_60MIN
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
            df_working['is_post_after_feed_60min'] = self.DEFAULT_IS_POST_AFTER_FEED_60MIN
            
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
                
                # Find records within 60 minutes after this feed
                within_window = (time_diffs >= 0) & (time_diffs <= 60)
                
                # Update the boolean feature for records within the window
                if within_window.any():
                    df_working.loc[within_window, 'is_post_after_feed_60min'] = True
            
            # Restore original index structure if needed
            if hist_was_indexed:
                df_working.set_index('created_at', inplace=True)
            
            return df_working
        
        except Exception as e:
            raise ValueError(f"Error preparing feed features: {e}")
    
    def detect_outliers(self,
                        data: pd.DataFrame,
                        column: str,
                        threshold: float =  2.0
                        ) -> pd.DataFrame:
        """
        Detect outliers in the specified column of the DataFrame using Z-score method.
        """
        try:
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in the DataFrame.")

            is_data_reindexed = False

            if not isinstance(data.index, pd.DatetimeIndex):
                 data = data.set_index('created_at')
                 is_data_reindexed = True

            temp_data = data.copy()
            temp_data = temp_data.dropna()
            temp_data = temp_data[[column]]

            if isinstance(temp_data.index, pd.DatetimeIndex):
                temp_data = temp_data.reset_index(drop=False, names='created_at')
        
            # Calculate Z-scores
            z_scores = np.abs(stats.zscore(temp_data[column])) # type: ignore
            outliers = np.where(z_scores > threshold)[0]

            # Mark outliers in the DataFrame
            temp_data['is_outlier'] = False
            temp_data.loc[outliers, 'is_outlier'] = True # type: ignore

            print(f"columns after outlier detection: {temp_data.columns}")

            temp_data = temp_data.set_index('created_at')

            if 'is_outlier' not in data.columns:
                data['is_outlier'] = False

            data.loc[temp_data.index, 'is_outlier'] = temp_data['is_outlier']

            if is_data_reindexed:
                data = data.reset_index(drop=False, names='created_at')

            return data
        except Exception as e:
            raise ValueError(f"Error detecting outliers: {e}")