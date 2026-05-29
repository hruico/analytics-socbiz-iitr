# =============================================================================
# tests/test_eda.py — EDA module unit tests
# =============================================================================

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.plots import (
    plot_demand_trend,
    plot_intraday_cycle,
    plot_weekday_weekend,
    plot_correlation_heatmap,
    plot_peak_volatility,
    plot_predicted_vs_actual,
    plot_reward_convergence,
    plot_theta_evolution,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_base_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "hourly_timestamp": timestamps,
        "acn_sessions_count": rng.integers(1, 20, n),
        "acn_total_kwh": rng.uniform(5.0, 200.0, n),
        "acn_base_revenue": rng.uniform(75.0, 3000.0, n),
        "urban_mean_utilization": rng.uniform(0.0, 1.0, n),
        "urban_peak_queue": rng.uniform(0.0, 20.0, n),
        "urban_total_volume": rng.uniform(10.0, 500.0, n),
        "hour_of_day": [t.hour for t in timestamps],
        "day_of_week": [t.dayofweek for t in timestamps],
        "is_weekend": [(1 if t.dayofweek >= 5 else 0) for t in timestamps],
    })


def _make_acn_df(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "connectionTime": pd.date_range("2023-01-01", periods=n, freq="30min"),
        "kWhDelivered": rng.uniform(1.0, 50.0, n),
        "stationID": [f"S{i % 10}" for i in range(n)],
        "connect_dt": pd.date_range("2023-01-01", periods=n, freq="30min"),
        "hourly_timestamp": pd.date_range("2023-01-01", periods=n, freq="30min").round("h"),
        "baseline_revenue": rng.uniform(15.0, 750.0, n),
    })


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: core plot functions create output files
# ─────────────────────────────────────────────────────────────────────────────

def test_plot_demand_trend_creates_file():
    df = _make_base_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_demand_trend(df, tmpdir)
        pngs = list(Path(tmpdir).glob("*.png"))
        assert len(pngs) >= 1, "plot_demand_trend did not create a PNG file"


def test_plot_intraday_cycle_creates_file():
    df = _make_base_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_intraday_cycle(df, tmpdir)
        pngs = list(Path(tmpdir).glob("*.png"))
        assert len(pngs) >= 1, "plot_intraday_cycle did not create a PNG file"


def test_plot_weekday_weekend_creates_file():
    df = _make_base_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_weekday_weekend(df, tmpdir)
        pngs = list(Path(tmpdir).glob("*.png"))
        assert len(pngs) >= 1, "plot_weekday_weekend did not create a PNG file"


def test_plot_correlation_heatmap_creates_file():
    df = _make_base_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_correlation_heatmap(df, tmpdir)
        pngs = list(Path(tmpdir).glob("*.png"))
        assert len(pngs) >= 1, "plot_correlation_heatmap did not create a PNG file"


def test_plot_peak_volatility_creates_file():
    df = _make_base_df()
    with tempfile.TemporaryDirectory() as tmpdir:
        plot_peak_volatility(df, tmpdir)
        pngs = list(Path(tmpdir).glob("*.png"))
        assert len(pngs) >= 1, "plot_peak_volatility did not create a PNG file"


# ─────────────────────────────────────────────────────────────────────────────
# Unit tests: post-run plots skip gracefully when files are absent
# ─────────────────────────────────────────────────────────────────────────────

def test_plot_predicted_vs_actual_skips_if_missing():
    """plot_predicted_vs_actual must not raise when predictions.csv is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_path = str(Path(tmpdir) / "predictions.csv")
        # Should log WARNING and return without raising
        plot_predicted_vs_actual(missing_path, tmpdir)
        # No PNG should be created
        pngs = list(Path(tmpdir).glob("*.png"))
        assert len(pngs) == 0


def test_plot_reward_convergence_skips_if_missing():
    """plot_reward_convergence must not raise when agentic_outcomes.csv is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_path = str(Path(tmpdir) / "agentic_outcomes.csv")
        plot_reward_convergence(missing_path, tmpdir)
        pngs = list(Path(tmpdir).glob("*.png"))
        assert len(pngs) == 0


def test_plot_theta_evolution_skips_if_missing():
    """plot_theta_evolution must not raise when agentic_outcomes.csv is absent."""
    with tempfile.TemporaryDirectory() as tmpdir:
        missing_path = str(Path(tmpdir) / "agentic_outcomes.csv")
        plot_theta_evolution(missing_path, tmpdir)
        pngs = list(Path(tmpdir).glob("*.png"))
        assert len(pngs) == 0


def test_plot_predicted_vs_actual_creates_file():
    """plot_predicted_vs_actual creates a PNG when predictions.csv exists."""
    rng = np.random.default_rng(42)
    n = 50
    df = pd.DataFrame({
        "actual_urban_mean_utilization": rng.uniform(0, 1, n),
        "pred_urban_mean_utilization": rng.uniform(0, 1, n),
        "actual_urban_peak_queue": rng.uniform(0, 20, n),
        "pred_urban_peak_queue": rng.uniform(0, 20, n),
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = str(Path(tmpdir) / "predictions.csv")
        df.to_csv(csv_path, index=False)
        plot_predicted_vs_actual(csv_path, tmpdir)
        pngs = list(Path(tmpdir).glob("*.png"))
        assert len(pngs) >= 1


def test_plot_reward_convergence_creates_file():
    """plot_reward_convergence creates a PNG when agentic_outcomes.csv exists."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({"reward": rng.uniform(-1, 1, n)})
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = str(Path(tmpdir) / "agentic_outcomes.csv")
        df.to_csv(csv_path, index=False)
        plot_reward_convergence(csv_path, tmpdir)
        pngs = list(Path(tmpdir).glob("*.png"))
        assert len(pngs) >= 1


def test_plot_theta_evolution_creates_file():
    """plot_theta_evolution creates a PNG when agentic_outcomes.csv has theta columns."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        "epsilon_after": rng.uniform(0.1, 5.0, n),
        "alpha_after": rng.uniform(1.0, 10.0, n),
        "beta_after": rng.uniform(1.0, 10.0, n),
    })
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = str(Path(tmpdir) / "agentic_outcomes.csv")
        df.to_csv(csv_path, index=False)
        plot_theta_evolution(csv_path, tmpdir)
        pngs = list(Path(tmpdir).glob("*.png"))
        assert len(pngs) >= 1
