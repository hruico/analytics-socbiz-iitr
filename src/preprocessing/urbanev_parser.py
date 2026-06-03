"""UrbanEV (ST-EVCDP) dataset parser with spatial features and hourly aggregation."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


class UrbanEVParser:
    """Parser for UrbanEV (ST-EVCDP) dataset with spatial feature preservation."""
    
    def __init__(self, avg_session_duration_hours: float = 1.5):
        """
        Initialize UrbanEV parser.
        
        Args:
            avg_session_duration_hours: Average session duration for queue calculation (default: 1.5 hours)
        """
        self.avg_session_duration_hours = avg_session_duration_hours
    
    def load_csv(self, csv_path: str) -> pd.DataFrame:
        """
        Load UrbanEV CSV file.
        
        Args:
            csv_path: Path to UrbanEV CSV file
            
        Returns:
            DataFrame with parsed data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
            ValueError: If required columns are missing
        """
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"UrbanEV CSV file not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        # Validate required columns
        required_columns = [
            'start_time', 'end_time', 'charging_volume',
            'waiting_time', 'station_utilization', 'station_id',
            'latitude', 'longitude'
        ]
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
    
    def normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize timestamps to UTC hourly format.
        
        Args:
            df: DataFrame with start_time and end_time columns
            
        Returns:
            DataFrame with normalized hourly_timestamp column
        """
        if df.empty:
            df['hourly_timestamp'] = pd.Series(dtype='datetime64[ns, UTC]')
            return df
        
        # Parse timestamps
        df['start_time'] = pd.to_datetime(df['start_time'], utc=True)
        df['end_time'] = pd.to_datetime(df['end_time'], utc=True)
        
        # Round start time to hour for aggregation
        df['hourly_timestamp'] = df['start_time'].dt.floor('h')
        
        return df
    
    def aggregate_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate to hourly granularity by station_id and timestamp.
        
        Args:
            df: DataFrame with normalized timestamps
            
        Returns:
            DataFrame with hourly aggregated metrics and spatial features
        """
        if df.empty:
            return pd.DataFrame(columns=[
                'hourly_timestamp', 'station_id', 'latitude', 'longitude',
                'urban_mean_utilization', 'urban_peak_queue', 'urban_total_volume'
            ])
        
        # Clip station_utilization to [0, 1] before aggregation
        df['station_utilization'] = df['station_utilization'].clip(0.0, 1.0)
        
        # Compute queue metric: waiting_time / avg_session_duration
        df['queue_metric'] = df['waiting_time'] / self.avg_session_duration_hours
        # Handle division by zero and negative values
        df['queue_metric'] = df['queue_metric'].fillna(0).clip(lower=0)
        
        # Group by hour and station
        grouped = df.groupby(['hourly_timestamp', 'station_id'])
        
        # Aggregate metrics
        aggregated = grouped.agg({
            'station_utilization': 'mean',  # Average utilization per hour
            'queue_metric': 'max',  # Peak queue per hour
            'charging_volume': 'sum',  # Total volume per hour
            'latitude': 'first',  # Preserve spatial coordinates
            'longitude': 'first'
        }).reset_index()
        
        # Rename columns to match schema
        aggregated.columns = [
            'hourly_timestamp', 'station_id', 'urban_mean_utilization',
            'urban_peak_queue', 'urban_total_volume', 'latitude', 'longitude'
        ]
        
        # Reorder columns to match spec
        aggregated = aggregated[[
            'hourly_timestamp', 'station_id', 'latitude', 'longitude',
            'urban_mean_utilization', 'urban_peak_queue', 'urban_total_volume'
        ]]
        
        # Convert timestamp to ISO format string
        aggregated['hourly_timestamp'] = aggregated['hourly_timestamp'].dt.strftime(
            '%Y-%m-%dT%H:%M:%S%z'
        )
        
        return aggregated
    
    def parse_and_export(
        self,
        csv_path: str,
        output_path: str = "data/processed/urbanev_hourly.csv"
    ) -> pd.DataFrame:
        """
        Parse UrbanEV CSV and export to processed CSV.
        
        Args:
            csv_path: Path to input CSV file
            output_path: Path to output CSV file
            
        Returns:
            Aggregated DataFrame
        """
        # Load CSV
        df = self.load_csv(csv_path)
        
        # Normalize timestamps
        df = self.normalize_timestamps(df)
        
        # Aggregate to hourly granularity
        aggregated = self.aggregate_hourly(df)
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        aggregated.to_csv(output_path, index=False)
        
        print(f"UrbanEV processed: {len(aggregated)} hourly records exported to {output_path}")
        
        return aggregated
