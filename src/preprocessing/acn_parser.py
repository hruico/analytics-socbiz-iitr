"""ACN-Data JSON to CSV parser with hourly aggregation."""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import numpy as np


class ACNDataParser:
    """Parser for ACN-Data JSON format with hourly aggregation."""
    
    def __init__(self, baseline_tariff: float = 15.0):
        """
        Initialize ACN parser.
        
        Args:
            baseline_tariff: Baseline tariff per kWh for revenue calculations (default: ₹15.0)
        """
        self.baseline_tariff = baseline_tariff
    
    def parse_json(self, json_path: str) -> pd.DataFrame:
        """
        Parse raw ACN-Data JSON file.
        
        Args:
            json_path: Path to ACN-Data JSON file
            
        Returns:
            DataFrame with parsed session data
            
        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON is malformed
            KeyError: If required fields are missing
        """
        json_path = Path(json_path)
        if not json_path.exists():
            raise FileNotFoundError(f"ACN-Data JSON file not found: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle both array of sessions and wrapped format
        sessions = data if isinstance(data, list) else data.get('sessions', [])
        
        if not sessions:
            # Return empty DataFrame with correct schema
            return pd.DataFrame(columns=[
                'connectionTime', 'disconnectTime', 'kWhDelivered',
                'sessionID', 'stationID'
            ])
        
        # Parse sessions
        parsed_sessions = []
        for session in sessions:
            try:
                parsed = {
                    'connectionTime': session['connectionTime'],
                    'disconnectTime': session['disconnectTime'],
                    'kWhDelivered': float(session['kWhDelivered']),
                    'sessionID': session['sessionID'],
                    'stationID': session.get('stationID', 'unknown')
                }
                parsed_sessions.append(parsed)
            except KeyError as e:
                raise KeyError(f"Missing required field in session: {e}")
        
        return pd.DataFrame(parsed_sessions)
    
    def normalize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize timestamps to UTC hourly format.
        
        Args:
            df: DataFrame with connectionTime and disconnectTime columns
            
        Returns:
            DataFrame with normalized hourly_timestamp column
        """
        if df.empty:
            df['hourly_timestamp'] = pd.Series(dtype='str')
            return df
        
        # Parse timestamps (handle various formats)
        df['connectionTime'] = pd.to_datetime(df['connectionTime'], utc=True)
        df['disconnectTime'] = pd.to_datetime(df['disconnectTime'], utc=True)
        
        # Round connection time to nearest hour for aggregation
        df['hourly_timestamp'] = df['connectionTime'].dt.floor('h')
        
        return df
    
    def aggregate_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate sessions to hourly granularity by stationID and timestamp.
        
        Args:
            df: DataFrame with normalized timestamps and session data
            
        Returns:
            DataFrame with hourly aggregated metrics
        """
        if df.empty:
            return pd.DataFrame(columns=[
                'hourly_timestamp', 'stationID', 'acn_sessions_count',
                'acn_total_kwh', 'acn_avg_kwh_per_session',
                'acn_base_revenue', 'acn_revenue_per_session',
                'acn_energy_cost_per_kwh'
            ])
        
        # Group by hourly timestamp and stationID
        grouped = df.groupby(['hourly_timestamp', 'stationID'])
        
        # Compute aggregated metrics
        aggregated = grouped.agg({
            'sessionID': 'count',  # Session count
            'kWhDelivered': ['sum', 'mean']  # Total and average kWh
        }).reset_index()
        
        # Flatten multi-level columns
        aggregated.columns = [
            'hourly_timestamp', 'stationID',
            'acn_sessions_count', 'acn_total_kwh', 'acn_avg_kwh_per_session'
        ]
        
        # Compute revenue metrics
        aggregated['acn_base_revenue'] = (
            aggregated['acn_total_kwh'] * self.baseline_tariff
        )
        aggregated['acn_revenue_per_session'] = (
            aggregated['acn_avg_kwh_per_session'] * self.baseline_tariff
        )
        
        # Energy cost per kWh (constant from config)
        aggregated['acn_energy_cost_per_kwh'] = self.baseline_tariff
        
        # Convert timestamp to ISO format string
        aggregated['hourly_timestamp'] = aggregated['hourly_timestamp'].dt.strftime(
            '%Y-%m-%dT%H:%M:%S%z'
        )
        
        return aggregated
    
    def parse_and_export(
        self,
        json_path: str,
        output_path: str = "data/processed/acn_hourly.csv"
    ) -> pd.DataFrame:
        """
        Parse ACN-Data JSON and export to CSV.
        
        Args:
            json_path: Path to input JSON file
            output_path: Path to output CSV file
            
        Returns:
            Aggregated DataFrame
        """
        # Parse JSON
        df = self.parse_json(json_path)
        
        # Normalize timestamps
        df = self.normalize_timestamps(df)
        
        # Aggregate to hourly granularity
        aggregated = self.aggregate_hourly(df)
        
        # Ensure output directory exists
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export to CSV
        aggregated.to_csv(output_path, index=False)
        
        print(f"ACN-Data processed: {len(aggregated)} hourly records exported to {output_path}")
        
        return aggregated
