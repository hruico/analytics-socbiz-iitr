"""Dataset fusion for ACN-Data and UrbanEV with spatial clustering."""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
from typing import Optional


class DatasetFusion:
    """Fuse ACN-Data and UrbanEV datasets with spatial clustering."""
    
    def __init__(self, n_clusters: int = 5, random_state: int = 42):
        """
        Initialize dataset fusion.
        
        Args:
            n_clusters: Number of clusters for spatial K-means (default: 5)
            random_state: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
    
    def load_datasets(
        self,
        acn_path: str = "data/processed/acn_hourly.csv",
        urbanev_path: str = "data/processed/urbanev_hourly.csv"
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load processed ACN and UrbanEV datasets.
        
        Args:
            acn_path: Path to ACN hourly CSV
            urbanev_path: Path to UrbanEV hourly CSV
            
        Returns:
            Tuple of (acn_df, urbanev_df)
            
        Raises:
            FileNotFoundError: If either file doesn't exist
        """
        acn_path = Path(acn_path)
        urbanev_path = Path(urbanev_path)
        
        if not acn_path.exists():
            raise FileNotFoundError(f"ACN hourly file not found: {acn_path}")
        if not urbanev_path.exists():
            raise FileNotFoundError(f"UrbanEV hourly file not found: {urbanev_path}")
        
        acn_df = pd.read_csv(acn_path)
        urbanev_df = pd.read_csv(urbanev_path)
        
        return acn_df, urbanev_df
    
    def align_timestamps(
        self,
        acn_df: pd.DataFrame,
        urbanev_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Align datasets by hourly timestamp using outer join.
        
        Args:
            acn_df: ACN hourly data
            urbanev_df: UrbanEV hourly data
            
        Returns:
            Merged DataFrame with all timestamps
        """
        # Ensure timestamp columns are present
        if 'hourly_timestamp' not in acn_df.columns:
            raise ValueError("ACN data missing 'hourly_timestamp' column")
        if 'hourly_timestamp' not in urbanev_df.columns:
            raise ValueError("UrbanEV data missing 'hourly_timestamp' column")
        
        # Map station IDs (handle different schemas)
        # For now, assume stationID in ACN and station_id in UrbanEV
        acn_df = acn_df.rename(columns={'stationID': 'station_id'})
        
        # Merge on hourly_timestamp (outer join to preserve all timestamps)
        merged = pd.merge(
            acn_df,
            urbanev_df,
            on=['hourly_timestamp', 'station_id'],
            how='outer',
            suffixes=('_acn', '_urbanev')
        )
        
        return merged
    
    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add temporal features: time_step, hour_of_day, day_of_week, is_weekend.
        
        Args:
            df: DataFrame with hourly_timestamp column
            
        Returns:
            DataFrame with temporal features added
        """
        # Parse timestamp
        df['timestamp_dt'] = pd.to_datetime(df['hourly_timestamp'], utc=True)
        
        # Sort chronologically
        df = df.sort_values('timestamp_dt').reset_index(drop=True)
        
        # Add time_step (sequential ordering)
        df['time_step'] = range(len(df))
        
        # Extract temporal features
        df['hour_of_day'] = df['timestamp_dt'].dt.hour
        df['day_of_week'] = df['timestamp_dt'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)  # 1 if Sat/Sun
        
        # Drop temporary column
        df = df.drop(columns=['timestamp_dt'])
        
        return df
    
    def apply_spatial_clustering(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply K-means clustering on station coordinates for spatial features.
        
        Args:
            df: DataFrame with latitude and longitude columns
            
        Returns:
            DataFrame with station_cluster_id feature added
        """
        # Check for spatial coordinates
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            # If missing, create default cluster
            df['station_cluster_id'] = 0
            return df
        
        # Get unique stations with coordinates
        stations = df[['station_id', 'latitude', 'longitude']].drop_duplicates()
        stations = stations.dropna(subset=['latitude', 'longitude'])
        
        if len(stations) < self.n_clusters:
            # Not enough stations for clustering
            df['station_cluster_id'] = 0
            return df
        
        # Perform K-means clustering
        coordinates = stations[['latitude', 'longitude']].values
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(coordinates)
        
        # Create mapping from station_id to cluster_id
        stations['station_cluster_id'] = cluster_labels
        station_to_cluster = dict(zip(stations['station_id'], stations['station_cluster_id']))
        
        # Map clusters to main dataframe
        df['station_cluster_id'] = df['station_id'].map(station_to_cluster).fillna(0).astype(int)
        
        return df
    
    def fuse_and_export(
        self,
        acn_path: str = "data/processed/acn_hourly.csv",
        urbanev_path: str = "data/processed/urbanev_hourly.csv",
        output_path: str = "data/processed/unified_analytical_base.csv"
    ) -> pd.DataFrame:
        """
        Fuse datasets and export to unified CSV.
        
        Args:
            acn_path: Path to ACN hourly CSV
            urbanev_path: Path to UrbanEV hourly CSV
            output_path: Path to output unified CSV
            
        Returns:
            Unified DataFrame
        """
        # Load datasets
        acn_df, urbanev_df = self.load_datasets(acn_path, urbanev_path)
        
        # Align by timestamp
        merged = self.align_timestamps(acn_df, urbanev_df)
        
        # Add temporal features
        merged = self.add_temporal_features(merged)
        
        # Apply spatial clustering
        merged = self.apply_spatial_clustering(merged)
        
        # Select and order final columns
        final_columns = [
            'hourly_timestamp', 'time_step',
            'acn_sessions_count', 'acn_total_kwh', 'acn_avg_kwh_per_session',
            'acn_base_revenue', 'acn_revenue_per_session', 'acn_energy_cost_per_kwh',
            'urban_mean_utilization', 'urban_peak_queue', 'urban_total_volume',
            'hour_of_day', 'day_of_week', 'is_weekend', 'station_cluster_id'
        ]
        
        # Only keep columns that exist
        existing_columns = [col for col in final_columns if col in merged.columns]
        unified = merged[existing_columns]
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        unified.to_csv(output_path, index=False)
        
        print(f"Dataset fusion complete: {len(unified)} rows exported to {output_path}")
        print(f"Spatial clustering: {unified['station_cluster_id'].nunique()} clusters created")
        
        return unified
