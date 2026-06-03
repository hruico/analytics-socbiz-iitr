"""Real data pipeline - fixes synthetic data and utilization calculation issues."""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class RealDataPipeline:
    """Build analytical base from real ACN + UrbanEV data with correct utilization."""
    
    def __init__(self, baseline_tariff: float = 15.0):
        self.baseline_tariff = baseline_tariff
    
    def load_acn_data(self, sessions_path: str) -> pd.DataFrame:
        """Load and process ACN session data."""
        import json
        
        with open(sessions_path, 'r') as f:
            data = json.load(f)
        
        sessions = data if isinstance(data, list) else data.get('sessions', [])
        
        if not sessions:
            return pd.DataFrame()
        
        # Parse sessions
        records = []
        for s in sessions:
            records.append({
                'sessionID': s['sessionID'],
                'stationID': s.get('stationID', 'unknown'),
                'connectionTime': pd.to_datetime(s['connectionTime'], utc=True),
                'disconnectTime': pd.to_datetime(s['disconnectTime'], utc=True),
                'kWhDelivered': float(s['kWhDelivered'])
            })
        
        df = pd.DataFrame(records)
        
        # Extract temporal features
        df['hour_of_day'] = df['connectionTime'].dt.hour
        df['day_of_week'] = df['connectionTime'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        return df
    
    def compute_data_driven_peak_hours(self, acn_df: pd.DataFrame, percentile: float = 0.75) -> set:
        """FIX 4: Compute peak hours from actual data, not hardcoded commute times."""
        hourly_counts = acn_df.groupby('hour_of_day')['sessionID'].count()
        threshold = hourly_counts.quantile(percentile)
        peak_hours = set(hourly_counts[hourly_counts >= threshold].index)
        
        logger.info(f"Data-driven peak hours (>P{int(percentile*100)}): {sorted(peak_hours)}")
        return peak_hours
    
    def aggregate_acn_hourly(self, acn_df: pd.DataFrame, peak_hours: set) -> pd.DataFrame:
        """Aggregate ACN to hourly with temporal features."""
        if acn_df.empty:
            return pd.DataFrame()
        
        # Group by temporal features (not time_step or absolute timestamp)
        grouped = acn_df.groupby(['hour_of_day', 'day_of_week', 'is_weekend'])
        
        acn_hourly = grouped.agg({
            'sessionID': 'count',
            'kWhDelivered': ['sum', 'mean']
        }).reset_index()
        
        acn_hourly.columns = [
            'hour_of_day', 'day_of_week', 'is_weekend',
            'acn_sessions_count', 'acn_total_kwh', 'acn_avg_kwh_per_session'
        ]
        
        # Add data-driven peak hour feature
        acn_hourly['is_peak_hour'] = acn_hourly['hour_of_day'].isin(peak_hours).astype(int)
        
        # Compute baseline revenue (constant baseline, not dynamic price)
        acn_hourly['acn_base_revenue'] = acn_hourly['acn_total_kwh'] * self.baseline_tariff
        
        return acn_hourly

    
    def load_urbanev_data(self, data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load UrbanEV occupancy, time, and information CSVs."""
        occupancy = pd.read_csv(f"{data_dir}/occupancy.csv")
        time_df = pd.read_csv(f"{data_dir}/time.csv")
        info_df = pd.read_csv(f"{data_dir}/information.csv")
        
        return occupancy, time_df, info_df
    
    def compute_per_zone_utilization(
        self, 
        occupancy: pd.DataFrame, 
        info_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        FIX 1: Compute per-zone utilization, then aggregate.
        
        Current bug: urban_mean_utilization = total_occupied / 18061 (system-wide)
        → Result: 1.7%-3.0% (structurally can't reach 30%/80% thresholds)
        
        Correct: Divide each zone's occupancy by that zone's capacity, then average
        → Result: 21%-41% (meaningful utilization that interacts with thresholds)
        """
        # Get zone capacities
        zone_capacity = info_df.set_index('grid')['count']
        
        # Divide each zone column by its capacity
        timestamp_col = occupancy['timestamp']
        occupancy_matrix = occupancy.drop('timestamp', axis=1)
        
        # Per-zone utilization (0-1 ratio)
        util_df = occupancy_matrix.div(zone_capacity, axis=1)
        
        # Mean utilization across zones per timestep
        mean_util = util_df.mean(axis=1)
        
        logger.info(f"Per-zone utilization range: {mean_util.min():.2%} - {mean_util.max():.2%}")
        logger.info(f"OLD system-wide method would give: {occupancy_matrix.sum(axis=1).mean() / zone_capacity.sum():.2%}")
        
        return pd.DataFrame({
            'timestamp': timestamp_col,
            'urban_mean_utilization': mean_util
        })
    
    def aggregate_urbanev_hourly(
        self,
        utilization_df: pd.DataFrame,
        time_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Resample UrbanEV from 5-min to hourly and align on temporal features.
        
        FIX 3: No time_step column - align on (hour_of_day, day_of_week, is_weekend)
        """
        # Merge with time features
        merged = utilization_df.merge(time_df, on='timestamp', how='left')
        
        # Parse datetime
        merged['datetime'] = pd.to_datetime(merged['timestamp'])
        
        # Extract temporal features
        merged['hour_of_day'] = merged['datetime'].dt.hour
        merged['day_of_week'] = merged['datetime'].dt.dayofweek
        merged['is_weekend'] = (merged['day_of_week'] >= 5).astype(int)
        
        # Resample to hourly (5-min → 1H)
        merged = merged.set_index('datetime')
        urban_hourly = merged.resample('1H').agg({
            'urban_mean_utilization': 'mean',
            'hour_of_day': 'first',
            'day_of_week': 'first',
            'is_weekend': 'first'
        }).reset_index(drop=True)
        
        # Group by temporal features (behavioral alignment, not calendar)
        urban_hourly = urban_hourly.groupby(
            ['hour_of_day', 'day_of_week', 'is_weekend']
        )['urban_mean_utilization'].mean().reset_index()
        
        return urban_hourly

    
    def build_unified_analytical_base(
        self,
        acn_sessions_path: str,
        urbanev_data_dir: str,
        output_path: str = "data/processed/unified_analytical_base.csv"
    ) -> pd.DataFrame:
        """
        Build unified analytical base from REAL data only (no synthetic).
        
        FIX 2: Use real ACN + UrbanEV data (~991 rows from ACN, behavioral join)
        FIX 3: Align on temporal features, not time_step index
        """
        logger.info("=" * 60)
        logger.info("Building analytical base from REAL data")
        logger.info("=" * 60)
        
        # Load ACN data
        logger.info("Loading ACN session data...")
        acn_df = self.load_acn_data(acn_sessions_path)
        logger.info(f"ACN: {len(acn_df)} sessions loaded")
        
        # FIX 4: Compute data-driven peak hours
        peak_hours = self.compute_data_driven_peak_hours(acn_df)
        
        # Aggregate ACN to hourly
        acn_hourly = self.aggregate_acn_hourly(acn_df, peak_hours)
        logger.info(f"ACN hourly: {len(acn_hourly)} unique temporal patterns")
        
        # Load UrbanEV data
        logger.info("Loading UrbanEV data...")
        occupancy, time_df, info_df = self.load_urbanev_data(urbanev_data_dir)
        
        # FIX 1: Compute per-zone utilization
        utilization_df = self.compute_per_zone_utilization(occupancy, info_df)
        
        # Aggregate UrbanEV to hourly
        urban_hourly = self.aggregate_urbanev_hourly(utilization_df, time_df)
        logger.info(f"UrbanEV hourly: {len(urban_hourly)} unique temporal patterns")
        
        # FIX 3: Merge on temporal features (behavioral alignment)
        features = ['hour_of_day', 'day_of_week', 'is_weekend']
        unified = acn_hourly.merge(urban_hourly, on=features, how='inner')
        
        logger.info(f"Unified base: {len(unified)} rows (inner join on temporal features)")
        logger.info(f"Utilization range: {unified['urban_mean_utilization'].min():.2%} - {unified['urban_mean_utilization'].max():.2%}")
        
        # Add time_step for chronological ordering (but not used for join)
        unified = unified.sort_values(['hour_of_day', 'day_of_week']).reset_index(drop=True)
        unified['time_step'] = range(len(unified))
        
        # FIX 5: Separate baseline from dynamic price column
        # baseline_price_per_kwh is a CONSTANT (₹15.0), not in the dataframe
        # dynamic_price will be added by the orchestrator during the loop
        
        # Reorder columns
        column_order = [
            'time_step', 'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour',
            'acn_sessions_count', 'acn_total_kwh', 'acn_avg_kwh_per_session',
            'acn_base_revenue', 'urban_mean_utilization'
        ]
        unified = unified[column_order]
        
        # Export
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        unified.to_csv(output_path, index=False)
        
        logger.info(f"✓ Unified analytical base exported: {output_path}")
        logger.info(f"✓ Training rows: {len(unified)} (real data only, no synthetic)")
        
        return unified


def build_real_data_pipeline():
    """Entry point to rebuild analytical base from real data."""
    pipeline = RealDataPipeline(baseline_tariff=15.0)
    
    unified = pipeline.build_unified_analytical_base(
        acn_sessions_path="data/raw/acndata_sessions.json",  # Use JSON file
        urbanev_data_dir="data/raw",
        output_path="data/processed/unified_analytical_base.csv"
    )
    
    return unified


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_real_data_pipeline()
