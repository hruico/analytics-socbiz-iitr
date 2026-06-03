"""Unit tests for ACN-Data parser."""
import pytest
import json
import pandas as pd
from pathlib import Path
import tempfile
from datetime import datetime, timezone

from src.preprocessing.acn_parser import ACNDataParser


@pytest.fixture
def sample_acn_data():
    """Sample ACN-Data JSON for testing."""
    return [
        {
            "sessionID": "session_001",
            "stationID": "station_A",
            "connectionTime": "2024-01-15T08:30:00Z",
            "disconnectTime": "2024-01-15T09:45:00Z",
            "kWhDelivered": 25.5
        },
        {
            "sessionID": "session_002",
            "stationID": "station_A",
            "connectionTime": "2024-01-15T08:45:00Z",
            "disconnectTime": "2024-01-15T10:15:00Z",
            "kWhDelivered": 18.2
        },
        {
            "sessionID": "session_003",
            "stationID": "station_B",
            "connectionTime": "2024-01-15T09:00:00Z",
            "disconnectTime": "2024-01-15T10:30:00Z",
            "kWhDelivered": 30.0
        }
    ]


@pytest.fixture
def temp_json_file(sample_acn_data):
    """Create a temporary JSON file with sample data."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_acn_data, f)
        temp_path = f.name
    yield temp_path
    Path(temp_path).unlink()


class TestACNDataParser:
    """Tests for ACNDataParser class."""
    
    def test_initialization_default_baseline(self):
        """Test parser initialization with default baseline tariff."""
        parser = ACNDataParser()
        assert parser.baseline_tariff == 15.0
    
    def test_initialization_custom_baseline(self):
        """Test parser initialization with custom baseline tariff."""
        parser = ACNDataParser(baseline_tariff=20.0)
        assert parser.baseline_tariff == 20.0
    
    def test_parse_json_valid_data(self, temp_json_file):
        """Test JSON parsing with valid session data."""
        parser = ACNDataParser()
        df = parser.parse_json(temp_json_file)
        
        assert len(df) == 3
        assert 'sessionID' in df.columns
        assert 'stationID' in df.columns
        assert 'connectionTime' in df.columns
        assert 'disconnectTime' in df.columns
        assert 'kWhDelivered' in df.columns
    
    def test_parse_json_missing_file(self):
        """Test JSON parsing with missing file."""
        parser = ACNDataParser()
        with pytest.raises(FileNotFoundError):
            parser.parse_json("nonexistent_file.json")
    
    def test_parse_json_empty_data(self):
        """Test JSON parsing with empty input."""
        parser = ACNDataParser()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump([], f)
            temp_path = f.name
        
        try:
            df = parser.parse_json(temp_path)
            assert len(df) == 0
            assert 'sessionID' in df.columns
        finally:
            Path(temp_path).unlink()
    
    def test_parse_json_missing_fields(self):
        """Test JSON parsing with missing required fields."""
        parser = ACNDataParser()
        incomplete_data = [
            {
                "sessionID": "session_001",
                # Missing kWhDelivered
                "connectionTime": "2024-01-15T08:30:00Z",
                "disconnectTime": "2024-01-15T09:45:00Z"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(incomplete_data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(KeyError):
                parser.parse_json(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_normalize_timestamps(self, temp_json_file):
        """Test timestamp normalization and UTC conversion."""
        parser = ACNDataParser()
        df = parser.parse_json(temp_json_file)
        df = parser.normalize_timestamps(df)
        
        assert 'hourly_timestamp' in df.columns
        assert pd.api.types.is_datetime64_any_dtype(df['hourly_timestamp'])
        # Check timestamps are rounded to hour
        assert all(df['hourly_timestamp'].dt.minute == 0)
        assert all(df['hourly_timestamp'].dt.second == 0)
    
    def test_aggregate_hourly(self, temp_json_file):
        """Test hourly aggregation correctness."""
        parser = ACNDataParser(baseline_tariff=15.0)
        df = parser.parse_json(temp_json_file)
        df = parser.normalize_timestamps(df)
        aggregated = parser.aggregate_hourly(df)
        
        # Check output columns
        expected_columns = [
            'hourly_timestamp', 'stationID', 'acn_sessions_count',
            'acn_total_kwh', 'acn_avg_kwh_per_session',
            'acn_base_revenue', 'acn_revenue_per_session',
            'acn_energy_cost_per_kwh'
        ]
        assert list(aggregated.columns) == expected_columns
        
        # Verify aggregation logic (2 sessions at station_A hour 08:00)
        station_a_hour_8 = aggregated[
            (aggregated['stationID'] == 'station_A') &
            (aggregated['hourly_timestamp'].str.contains('2024-01-15T08:'))
        ]
        
        if len(station_a_hour_8) > 0:
            assert station_a_hour_8['acn_sessions_count'].iloc[0] == 2
            # Total kWh should be 25.5 + 18.2 = 43.7
            assert abs(station_a_hour_8['acn_total_kwh'].iloc[0] - 43.7) < 0.01
            # Average kWh should be 43.7 / 2 = 21.85
            assert abs(station_a_hour_8['acn_avg_kwh_per_session'].iloc[0] - 21.85) < 0.01
    
    def test_revenue_calculations(self, temp_json_file):
        """Test revenue metric calculations."""
        baseline = 20.0
        parser = ACNDataParser(baseline_tariff=baseline)
        df = parser.parse_json(temp_json_file)
        df = parser.normalize_timestamps(df)
        aggregated = parser.aggregate_hourly(df)
        
        # Check revenue calculations
        for _, row in aggregated.iterrows():
            expected_base_revenue = row['acn_total_kwh'] * baseline
            expected_revenue_per_session = row['acn_avg_kwh_per_session'] * baseline
            
            assert abs(row['acn_base_revenue'] - expected_base_revenue) < 0.01
            assert abs(row['acn_revenue_per_session'] - expected_revenue_per_session) < 0.01
            assert row['acn_energy_cost_per_kwh'] == baseline
    
    def test_parse_and_export(self, temp_json_file):
        """Test full pipeline with export."""
        parser = ACNDataParser()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "acn_hourly.csv"
            result = parser.parse_and_export(temp_json_file, str(output_path))
            
            # Check file was created
            assert output_path.exists()
            
            # Check CSV can be read back
            df_from_csv = pd.read_csv(output_path)
            assert len(df_from_csv) > 0
            assert 'acn_sessions_count' in df_from_csv.columns
    
    def test_empty_dataframe_handling(self):
        """Test handling of empty DataFrame in aggregation."""
        parser = ACNDataParser()
        empty_df = pd.DataFrame(columns=[
            'connectionTime', 'disconnectTime', 'kWhDelivered',
            'sessionID', 'stationID'
        ])
        
        # Should not crash
        normalized = parser.normalize_timestamps(empty_df)
        aggregated = parser.aggregate_hourly(normalized)
        
        assert len(aggregated) == 0
        assert 'acn_sessions_count' in aggregated.columns
    
    def test_multiple_sessions_per_hour(self):
        """Test JSON parsing with multiple sessions per hour (explicit aggregation test)."""
        # Create data with 3 sessions in the same hour at the same station
        data = [
            {
                "sessionID": "session_001",
                "stationID": "station_X",
                "connectionTime": "2024-01-15T14:10:00Z",
                "disconnectTime": "2024-01-15T15:00:00Z",
                "kWhDelivered": 10.0
            },
            {
                "sessionID": "session_002",
                "stationID": "station_X",
                "connectionTime": "2024-01-15T14:30:00Z",
                "disconnectTime": "2024-01-15T15:30:00Z",
                "kWhDelivered": 15.0
            },
            {
                "sessionID": "session_003",
                "stationID": "station_X",
                "connectionTime": "2024-01-15T14:45:00Z",
                "disconnectTime": "2024-01-15T16:00:00Z",
                "kWhDelivered": 20.0
            }
        ]
        
        parser = ACNDataParser(baseline_tariff=15.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            df = parser.parse_json(temp_path)
            df = parser.normalize_timestamps(df)
            aggregated = parser.aggregate_hourly(df)
            
            # All 3 sessions should be aggregated into the 14:00 hour
            assert len(aggregated) == 1
            assert aggregated['acn_sessions_count'].iloc[0] == 3
            assert abs(aggregated['acn_total_kwh'].iloc[0] - 45.0) < 0.01  # 10 + 15 + 20
            assert abs(aggregated['acn_avg_kwh_per_session'].iloc[0] - 15.0) < 0.01  # 45 / 3
            assert abs(aggregated['acn_base_revenue'].iloc[0] - 675.0) < 0.01  # 45 * 15
            assert abs(aggregated['acn_revenue_per_session'].iloc[0] - 225.0) < 0.01  # 15 * 15
        finally:
            Path(temp_path).unlink()
    
    def test_missing_sessionID_field(self):
        """Test handling of missing sessionID in JSON."""
        data = [
            {
                "stationID": "station_A",
                "connectionTime": "2024-01-15T08:30:00Z",
                "disconnectTime": "2024-01-15T09:45:00Z",
                "kWhDelivered": 25.5
                # Missing sessionID
            }
        ]
        
        parser = ACNDataParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(KeyError, match="sessionID"):
                parser.parse_json(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_missing_kWhDelivered_field(self):
        """Test handling of missing kWhDelivered in JSON."""
        data = [
            {
                "sessionID": "session_001",
                "stationID": "station_A",
                "connectionTime": "2024-01-15T08:30:00Z",
                "disconnectTime": "2024-01-15T09:45:00Z"
                # Missing kWhDelivered
            }
        ]
        
        parser = ACNDataParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(KeyError, match="kWhDelivered"):
                parser.parse_json(temp_path)
        finally:
            Path(temp_path).unlink()
    
    def test_timestamp_utc_conversion(self):
        """Test timestamp normalization handles various time zones correctly."""
        # Create data with non-UTC timezone
        data = [
            {
                "sessionID": "session_001",
                "stationID": "station_A",
                "connectionTime": "2024-01-15T08:30:00-05:00",  # EST
                "disconnectTime": "2024-01-15T09:45:00-05:00",
                "kWhDelivered": 25.5
            }
        ]
        
        parser = ACNDataParser()
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            df = parser.parse_json(temp_path)
            df = parser.normalize_timestamps(df)
            
            # Verify timestamps are in UTC
            assert df['connectionTime'].dt.tz is not None
            assert str(df['connectionTime'].dt.tz) == 'UTC'
            
            # Verify hourly rounding works correctly
            assert df['hourly_timestamp'].iloc[0].minute == 0
            assert df['hourly_timestamp'].iloc[0].second == 0
        finally:
            Path(temp_path).unlink()
    
    def test_hourly_aggregation_sums(self):
        """Test hourly aggregation correctness with explicit verification of sums."""
        data = [
            {
                "sessionID": "session_001",
                "stationID": "station_A",
                "connectionTime": "2024-01-15T08:15:00Z",
                "disconnectTime": "2024-01-15T09:00:00Z",
                "kWhDelivered": 10.5
            },
            {
                "sessionID": "session_002",
                "stationID": "station_A",
                "connectionTime": "2024-01-15T08:45:00Z",
                "disconnectTime": "2024-01-15T10:00:00Z",
                "kWhDelivered": 20.3
            },
            {
                "sessionID": "session_003",
                "stationID": "station_B",
                "connectionTime": "2024-01-15T08:30:00Z",
                "disconnectTime": "2024-01-15T09:30:00Z",
                "kWhDelivered": 15.7
            }
        ]
        
        parser = ACNDataParser(baseline_tariff=18.0)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = f.name
        
        try:
            df = parser.parse_json(temp_path)
            df = parser.normalize_timestamps(df)
            aggregated = parser.aggregate_hourly(df)
            
            # Should have 2 rows: station_A hour 08, station_B hour 08
            assert len(aggregated) == 2
            
            # Check station_A aggregation
            station_a = aggregated[aggregated['stationID'] == 'station_A']
            assert len(station_a) == 1
            assert station_a['acn_sessions_count'].iloc[0] == 2
            assert abs(station_a['acn_total_kwh'].iloc[0] - 30.8) < 0.01  # 10.5 + 20.3
            assert abs(station_a['acn_avg_kwh_per_session'].iloc[0] - 15.4) < 0.01  # 30.8 / 2
            
            # Check station_B aggregation
            station_b = aggregated[aggregated['stationID'] == 'station_B']
            assert len(station_b) == 1
            assert station_b['acn_sessions_count'].iloc[0] == 1
            assert abs(station_b['acn_total_kwh'].iloc[0] - 15.7) < 0.01
            assert abs(station_b['acn_avg_kwh_per_session'].iloc[0] - 15.7) < 0.01
        finally:
            Path(temp_path).unlink()
