"""Unit tests for UrbanEV parser."""
import pytest
import pandas as pd
from pathlib import Path
import tempfile

from src.preprocessing.urbanev_parser import UrbanEVParser


@pytest.fixture
def sample_urbanev_data():
    """Sample UrbanEV CSV data for testing."""
    return pd.DataFrame([
        {
            "station_id": "station_001",
            "start_time": "2024-01-15T08:15:00Z",
            "end_time": "2024-01-15T09:30:00Z",
            "charging_volume": 25.5,
            "waiting_time": 0.5,  # hours
            "station_utilization": 0.75,
            "latitude": 40.7128,
            "longitude": -74.0060
        },
        {
            "station_id": "station_001",
            "start_time": "2024-01-15T08:45:00Z",
            "end_time": "2024-01-15T10:00:00Z",
            "charging_volume": 18.2,
            "waiting_time": 1.2,
            "station_utilization": 0.82,
            "latitude": 40.7128,
            "longitude": -74.0060
        },
        {
            "station_id": "station_002",
            "start_time": "2024-01-15T09:00:00Z",
            "end_time": "2024-01-15T10:30:00Z",
            "charging_volume": 30.0,
            "waiting_time": 0.3,
            "station_utilization": 0.60,
            "latitude": 40.7580,
            "longitude": -73.9855
        }
    ])


@pytest.fixture
def temp_csv_file(sample_urbanev_data):
    """Create a temporary CSV file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_urbanev_data.to_csv(f.name, index=False)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink()


class TestUrbanEVParser:
    """Tests for UrbanEVParser class."""
    
    def test_initialization_default_duration(self):
        """Test parser initialization with default session duration."""
        parser = UrbanEVParser()
        assert parser.avg_session_duration_hours == 1.5
    
    def test_initialization_custom_duration(self):
        """Test parser initialization with custom session duration."""
        parser = UrbanEVParser(avg_session_duration_hours=2.0)
        assert parser.avg_session_duration_hours == 2.0
    
    def test_load_csv_valid_data(self, temp_csv_file):
        """Test CSV loading with valid data and schema validation."""
        parser = UrbanEVParser()
        df = parser.load_csv(temp_csv_file)
        
        assert len(df) == 3
        assert 'station_id' in df.columns
        assert 'start_time' in df.columns
        assert 'charging_volume' in df.columns
        assert 'latitude' in df.columns
        assert 'longitude' in df.columns
    
    def test_load_csv_missing_file(self):
        """Test CSV loading with missing file."""
        parser = UrbanEVParser()
        with pytest.raises(FileNotFoundError):
            parser.load_csv("nonexistent_file.csv")
    
    def test_load_csv_missing_columns(self):
        """Test CSV loading with missing required columns."""
        parser = UrbanEVParser()
        
        # Create CSV with missing columns
        incomplete_data = pd.DataFrame([
            {
                "station_id": "station_001",
                "start_time": "2024-01-15T08:15:00Z",
                # Missing charging_volume, waiting_time, etc.
            }
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            incomplete_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Missing required columns"):
                parser.load_csv(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_normalize_timestamps(self, temp_csv_file):
        """Test timestamp normalization to UTC hourly format."""
        parser = UrbanEVParser()
        df = parser.load_csv(temp_csv_file)
        df = parser.normalize_timestamps(df)
        
        assert 'hourly_timestamp' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['hourly_timestamp'])
        # Check timestamps are rounded to hour
        assert all(df['hourly_timestamp'].dt.minute == 0)
        assert all(df['hourly_timestamp'].dt.second == 0)
    
    def test_aggregate_hourly(self, temp_csv_file):
        """Test hourly aggregation across multiple stations."""
        parser = UrbanEVParser(avg_session_duration_hours=1.5)
        df = parser.load_csv(temp_csv_file)
        df = parser.normalize_timestamps(df)
        aggregated = parser.aggregate_hourly(df)
        
        # Check output columns
        expected_columns = [
            'hourly_timestamp', 'station_id', 'latitude', 'longitude',
            'urban_mean_utilization', 'urban_peak_queue', 'urban_total_volume'
        ]
        assert list(aggregated.columns) == expected_columns
        
        # Verify aggregation for station_001 hour 08:00 (2 sessions)
        station_1_hour_8 = aggregated[
            (aggregated['station_id'] == 'station_001') &
            (aggregated['hourly_timestamp'].str.contains('2024-01-15T08:'))
        ]
        
        if len(station_1_hour_8) > 0:
            # Mean utilization = (0.75 + 0.82) / 2 = 0.785
            assert abs(station_1_hour_8['urban_mean_utilization'].iloc[0] - 0.785) < 0.01
            # Peak queue = max(0.5/1.5, 1.2/1.5) = max(0.333, 0.8) = 0.8
            assert abs(station_1_hour_8['urban_peak_queue'].iloc[0] - 0.8) < 0.01
            # Total volume = 25.5 + 18.2 = 43.7
            assert abs(station_1_hour_8['urban_total_volume'].iloc[0] - 43.7) < 0.01
    
    def test_utilization_clipping(self):
        """Test utilization clipping to [0, 1] range."""
        parser = UrbanEVParser()
        
        # Create data with out-of-range utilization
        data = pd.DataFrame([
            {
                "station_id": "station_001",
                "start_time": "2024-01-15T08:00:00Z",
                "end_time": "2024-01-15T09:00:00Z",
                "charging_volume": 20.0,
                "waiting_time": 0.5,
                "station_utilization": 1.5,  # Over 1.0
                "latitude": 40.7128,
                "longitude": -74.0060
            },
            {
                "station_id": "station_002",
                "start_time": "2024-01-15T08:00:00Z",
                "end_time": "2024-01-15T09:00:00Z",
                "charging_volume": 15.0,
                "waiting_time": 0.3,
                "station_utilization": -0.2,  # Below 0.0
                "latitude": 40.7580,
                "longitude": -73.9855
            }
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            df = parser.load_csv(temp_path)
            df = parser.normalize_timestamps(df)
            aggregated = parser.aggregate_hourly(df)
            
            # All utilization values should be clipped to [0, 1]
            assert all(aggregated['urban_mean_utilization'] >= 0.0)
            assert all(aggregated['urban_mean_utilization'] <= 1.0)
        finally:
            Path(temp_path).unlink()
    
    def test_queue_calculation_zero_duration(self):
        """Test queue calculation with zero session duration."""
        parser = UrbanEVParser(avg_session_duration_hours=0.0)  # Edge case
        
        data = pd.DataFrame([
            {
                "station_id": "station_001",
                "start_time": "2024-01-15T08:00:00Z",
                "end_time": "2024-01-15T09:00:00Z",
                "charging_volume": 20.0,
                "waiting_time": 1.0,
                "station_utilization": 0.75,
                "latitude": 40.7128,
                "longitude": -74.0060
            }
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            df = parser.load_csv(temp_path)
            df = parser.normalize_timestamps(df)
            aggregated = parser.aggregate_hourly(df)
            
            # Should handle division by zero gracefully (fillna(0))
            assert not aggregated['urban_peak_queue'].isna().any()
            assert all(aggregated['urban_peak_queue'] >= 0)
        finally:
            Path(temp_path).unlink()
    
    def test_spatial_coordinates_preservation(self, temp_csv_file):
        """Test preservation of spatial coordinates (latitude, longitude)."""
        parser = UrbanEVParser()
        df = parser.load_csv(temp_csv_file)
        df = parser.normalize_timestamps(df)
        aggregated = parser.aggregate_hourly(df)
        
        # Check latitude and longitude are preserved
        assert 'latitude' in aggregated.columns
        assert 'longitude' in aggregated.columns
        assert not aggregated['latitude'].isna().any()
        assert not aggregated['longitude'].isna().any()
        
        # Verify specific coordinates for station_001
        station_1 = aggregated[aggregated['station_id'] == 'station_001']
        if len(station_1) > 0:
            assert abs(station_1['latitude'].iloc[0] - 40.7128) < 0.0001
            assert abs(station_1['longitude'].iloc[0] + 74.0060) < 0.0001
    
    def test_parse_and_export(self, temp_csv_file):
        """Test full pipeline with export."""
        parser = UrbanEVParser()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "urbanev_hourly.csv"
            result = parser.parse_and_export(temp_csv_file, str(output_path))
            
            # Check file was created
            assert output_path.exists()
            
            # Check CSV can be read back
            df_from_csv = pd.read_csv(output_path)
            assert len(df_from_csv) > 0
            assert 'urban_mean_utilization' in df_from_csv.columns
            assert 'latitude' in df_from_csv.columns
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame."""
        parser = UrbanEVParser()
        empty_df = pd.DataFrame(columns=[
            'start_time', 'end_time', 'charging_volume',
            'waiting_time', 'station_utilization', 'station_id',
            'latitude', 'longitude'
        ])
        
        # Should not crash
        normalized = parser.normalize_timestamps(empty_df)
        aggregated = parser.aggregate_hourly(normalized)
        
        assert len(aggregated) == 0
        assert 'urban_mean_utilization' in aggregated.columns
    
    def test_multiple_stations_aggregation(self):
        """Test aggregation across multiple stations (simulating 24,798 charging piles)."""
        parser = UrbanEVParser()
        
        # Create data with multiple stations and hours
        data = pd.DataFrame([
            {
                "station_id": f"station_{i:03d}",
                "start_time": "2024-01-15T08:00:00Z",
                "end_time": "2024-01-15T09:00:00Z",
                "charging_volume": 10.0 + i,
                "waiting_time": 0.5,
                "station_utilization": 0.6 + (i * 0.01),
                "latitude": 40.0 + (i * 0.001),
                "longitude": -74.0 + (i * 0.001)
            }
            for i in range(10)
        ])
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            df = parser.load_csv(temp_path)
            df = parser.normalize_timestamps(df)
            aggregated = parser.aggregate_hourly(df)
            
            # Should have 10 rows (one per station for hour 08:00)
            assert len(aggregated) == 10
            
            # Each station should have unique coordinates
            unique_stations = aggregated['station_id'].unique()
            assert len(unique_stations) == 10
        finally:
            Path(temp_path).unlink()
