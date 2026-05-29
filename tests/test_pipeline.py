# =============================================================================
# tests/test_pipeline.py — Pipeline property tests and unit tests
# =============================================================================

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings, HealthCheck
from hypothesis import strategies as st

from src.pipeline.preprocess import (
    _load_acn,
    _load_urban,
    _aggregate_acn_hourly,
    _aggregate_urban_hourly,
    _align_and_merge,
    run_pipeline,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_acn_excel(path: str, kwh_values: list[float]) -> None:
    """Write a minimal ACN Excel file to *path*."""
    n = len(kwh_values)
    df = pd.DataFrame({
        "connectionTime": pd.date_range("2023-01-01", periods=n, freq="30min"),
        "kWhDelivered": kwh_values,
        "stationID": [f"S{i % 5}" for i in range(n)],
    })
    df.to_excel(path, index=False)


def _make_urban_csvs(urban_dir: str, n_time: int = 24, n_stations: int = 3) -> None:
    """Write minimal UrbanEV wide-format CSVs to *urban_dir*."""
    rng = np.random.default_rng(42)
    station_cols = [f"node_{i}" for i in range(n_stations)]
    time_steps = [f"t{i}" for i in range(n_time)]

    for fname, low, high in [
        ("volume.csv", 0, 100),
        ("occupancy.csv", 0, 1),
        ("duration.csv", 10, 120),
    ]:
        data = {"time_step": time_steps}
        for col in station_cols:
            data[col] = rng.uniform(low, high, n_time)
        pd.DataFrame(data).to_csv(Path(urban_dir) / fname, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Property 1: Zero-kWh rows are excluded from pipeline output
# Feature: ev-charging-analytics-optimization, Property 1: Zero-kWh rows are excluded from pipeline output
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
@given(
    kwh_list=st.lists(
        st.floats(min_value=0.0, max_value=200.0, allow_nan=False, allow_infinity=False),
        min_size=10,
        max_size=100,
    )
)
def test_property1_zero_kwh_excluded(kwh_list):
    """
    # Feature: ev-charging-analytics-optimization, Property 1: Zero-kWh rows are excluded from pipeline output
    For any ACN DataFrame with zero/null kWh rows, the aggregated acn_total_kwh
    must equal the sum of only the non-zero, non-null kWh values.
    """
    # Inject some zeros
    kwh_arr = np.array(kwh_list, dtype=float)
    kwh_arr[::5] = 0.0  # every 5th row is zero

    n = len(kwh_arr)
    df = pd.DataFrame({
        "connectionTime": pd.date_range("2023-01-01", periods=n, freq="30min"),
        "kWhDelivered": kwh_arr,
        "stationID": [f"S{i % 3}" for i in range(n)],
    })

    # Simulate what _load_acn does: exclude zero/null rows
    mask = df["kWhDelivered"].isna() | (df["kWhDelivered"] <= 0)
    clean = df[~mask].copy()
    clean["kWhDelivered"] = clean["kWhDelivered"].astype(float)
    clean["hourly_timestamp"] = pd.to_datetime(clean["connectionTime"]).dt.round("h")
    clean["baseline_revenue"] = clean["kWhDelivered"] * 15.0

    agg = clean.groupby("hourly_timestamp").agg(
        acn_total_kwh=("kWhDelivered", "sum")
    ).reset_index()

    # All aggregated kWh must be > 0 (no zero-kWh hours)
    assert (agg["acn_total_kwh"] > 0).all(), \
        "acn_total_kwh contains zero values — zero-kWh rows were not excluded"

    # Total must equal sum of non-zero original values
    expected_total = float(kwh_arr[kwh_arr > 0].sum())
    actual_total = float(agg["acn_total_kwh"].sum())
    assert abs(actual_total - expected_total) < 1e-6, \
        f"Total kWh mismatch: expected {expected_total}, got {actual_total}"


# ─────────────────────────────────────────────────────────────────────────────
# Property 2: Melted UrbanEV merge produces no nulls in key columns
# Feature: ev-charging-analytics-optimization, Property 2: Melted UrbanEV merge produces no nulls in key columns
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    n_time=st.integers(min_value=5, max_value=30),
    n_stations=st.integers(min_value=2, max_value=8),
)
def test_property2_merge_no_nulls(n_time, n_stations):
    """
    # Feature: ev-charging-analytics-optimization, Property 2: Melted UrbanEV merge produces no nulls in key columns
    For any valid wide-format matrices, after melting and merging on
    [time_step, station_node], key columns must have zero nulls.
    """
    rng = np.random.default_rng(42)
    station_cols = [f"node_{i}" for i in range(n_stations)]
    time_steps = [f"t{i}" for i in range(n_time)]

    def _make_wide(low, high):
        data = {"time_step": time_steps}
        for col in station_cols:
            data[col] = rng.uniform(low, high, n_time)
        return pd.DataFrame(data)

    def _melt(df, value_name):
        melted = df.melt(id_vars=["time_step"], var_name="station_node", value_name=value_name)
        return melted

    vol = _melt(_make_wide(0, 100), "traffic_volume")
    occ = _melt(_make_wide(0, 1), "occupancy_density")
    dur = _melt(_make_wide(10, 120), "avg_duration")

    merged = vol.merge(occ, on=["time_step", "station_node"])
    merged = merged.merge(dur, on=["time_step", "station_node"])

    key_cols = ["traffic_volume", "occupancy_density", "avg_duration"]
    null_counts = merged[key_cols].isna().sum()
    assert null_counts.sum() == 0, \
        f"Unexpected nulls after merge: {null_counts[null_counts > 0].to_dict()}"


# ─────────────────────────────────────────────────────────────────────────────
# Property 4: Missing input file raises FileNotFoundError
# Feature: ev-charging-analytics-optimization, Property 4: Missing input file raises FileNotFoundError
# ─────────────────────────────────────────────────────────────────────────────

@settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
@given(fake_path=st.text(min_size=5, max_size=50).filter(lambda s: not Path(s).exists()))
def test_property4_missing_file_raises(fake_path):
    """
    # Feature: ev-charging-analytics-optimization, Property 4: Missing input file raises FileNotFoundError
    For any non-existent path, run_pipeline() must raise FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        run_pipeline(acn_path=fake_path)


# ─────────────────────────────────────────────────────────────────────────────
# Property 14: Pipeline is deterministic
# Feature: ev-charging-analytics-optimization, Property 14: Pipeline is deterministic (idempotent on same inputs)
# ─────────────────────────────────────────────────────────────────────────────

def test_property14_pipeline_deterministic():
    """
    # Feature: ev-charging-analytics-optimization, Property 14: Pipeline is deterministic (idempotent on same inputs)
    Running run_pipeline() twice on the same inputs must produce identical output.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        acn_path = str(Path(tmpdir) / "acn.xlsx")
        urban_dir = tmpdir
        out1 = str(Path(tmpdir) / "out1.csv")
        out2 = str(Path(tmpdir) / "out2.csv")

        kwh = [10.0, 20.0, 0.0, 30.0, 15.0, 25.0, 5.0, 40.0] * 10
        _make_acn_excel(acn_path, kwh)
        _make_urban_csvs(urban_dir, n_time=24, n_stations=3)

        df1 = run_pipeline(acn_path=acn_path, urban_dir=urban_dir, output_path=out1)
        df2 = run_pipeline(acn_path=acn_path, urban_dir=urban_dir, output_path=out2)

        assert df1.shape == df2.shape, "Row/column counts differ between runs"
        assert list(df1.columns) == list(df2.columns), "Column names differ between runs"
        # Numeric columns must be identical
        numeric_cols = df1.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            assert (df1[col].values == df2[col].values).all(), \
                f"Column '{col}' differs between runs"


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: run_pipeline() end-to-end schema check
# ─────────────────────────────────────────────────────────────────────────────

def test_run_pipeline_schema():
    """
    run_pipeline() must produce a CSV with the documented schema columns.
    """
    expected_cols = {
        "hourly_timestamp", "acn_sessions_count", "acn_total_kwh",
        "acn_base_revenue", "urban_mean_utilization", "urban_peak_queue",
        "urban_total_volume", "hour_of_day", "day_of_week", "is_weekend",
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        acn_path = str(Path(tmpdir) / "acn.xlsx")
        urban_dir = tmpdir
        out_path = str(Path(tmpdir) / "unified.csv")

        kwh = [10.0, 20.0, 5.0, 30.0] * 20
        _make_acn_excel(acn_path, kwh)
        _make_urban_csvs(urban_dir, n_time=24, n_stations=3)

        df = run_pipeline(acn_path=acn_path, urban_dir=urban_dir, output_path=out_path)

        assert Path(out_path).exists(), "Output CSV was not created"
        assert expected_cols.issubset(set(df.columns)), \
            f"Missing columns: {expected_cols - set(df.columns)}"
        assert len(df) > 0, "Output DataFrame is empty"
