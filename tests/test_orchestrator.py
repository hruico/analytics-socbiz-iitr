# =============================================================================
# tests/test_orchestrator.py — Orchestrator CLI and initialisation tests
# =============================================================================

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from orchestrator import parse_args


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_unified_csv(path: str, n: int = 300) -> None:
    rng = np.random.default_rng(42)
    timestamps = pd.date_range("2023-01-01", periods=n, freq="h")
    df = pd.DataFrame({
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
    df.to_csv(path, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: parse_args() accepts all documented arguments
# ─────────────────────────────────────────────────────────────────────────────

def test_parse_args_all_arguments():
    """parse_args() must accept all documented CLI arguments."""
    import sys
    test_argv = [
        "--csv", "data/processed/unified_analytical_base.csv",
        "--steps", "100",
        "--verbose", "10",
        "--lr", "0.8",
        "--decay", "0.002",
        "--delay", "1.0",
        "--epsilon", "1.2",
        "--alpha", "4.0",
        "--beta", "4.0",
        "--out", "outputs/agentic_outcomes.csv",
        "--predictions", "outputs/predictions.csv",
        "--log-level", "INFO",
        "--lightgbm",
    ]
    with patch("sys.argv", ["orchestrator.py"] + test_argv):
        args = parse_args()

    assert args.csv == "data/processed/unified_analytical_base.csv"
    assert args.steps == 100
    assert args.verbose == 10
    assert args.lr == 0.8
    assert args.decay == 0.002
    assert args.delay == 1.0
    assert args.epsilon == 1.2
    assert args.alpha == 4.0
    assert args.beta == 4.0
    assert args.out == "outputs/agentic_outcomes.csv"
    assert args.predictions == "outputs/predictions.csv"
    assert args.log_level == "INFO"
    assert args.lightgbm is True


def test_parse_args_defaults():
    """parse_args() must have sensible defaults."""
    with patch("sys.argv", ["orchestrator.py"]):
        args = parse_args()
    assert args.steps is None
    assert args.lightgbm is False
    assert args.log_level == "INFO"


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: AgenticOrchestrator raises FileNotFoundError for missing CSV
# ─────────────────────────────────────────────────────────────────────────────

def test_orchestrator_missing_csv_raises(monkeypatch):
    """AgenticOrchestrator must raise FileNotFoundError when CSV path is missing."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    from orchestrator import AgenticOrchestrator
    with pytest.raises(FileNotFoundError, match="not found"):
        AgenticOrchestrator(csv_path="/nonexistent/path/unified.csv")


# ─────────────────────────────────────────────────────────────────────────────
# Unit test: export_predictions() schema
# ─────────────────────────────────────────────────────────────────────────────

def test_export_predictions_schema(monkeypatch):
    """export_predictions() must produce CSV with documented column names."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    monkeypatch.setattr("src.agents.demand.XGB_PARAMS", {
        "n_estimators": 20, "learning_rate": 0.1, "max_depth": 3,
        "n_jobs": 1, "verbosity": 0, "tree_method": "hist",
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = str(Path(tmpdir) / "unified.csv")
        _make_unified_csv(csv_path, n=300)
        out_path = str(Path(tmpdir) / "predictions.csv")

        with patch("src.agents.pricing.build_gemini_model"), \
             patch("src.agents.monitoring.build_gemini_model"):

            from orchestrator import AgenticOrchestrator
            orch = AgenticOrchestrator.__new__(AgenticOrchestrator)
            orch._csv_path = csv_path
            orch._api_delay = 0.0
            orch._use_lightgbm = False

            from src.agents.demand import DemandPredictionAgent
            from src.agents.pricing import TariffPricingAgent
            from src.agents.monitoring import MonitoringLearningAgent
            import numpy as np

            orch.demand_agent = DemandPredictionAgent(csv_path=csv_path)

            pricing = TariffPricingAgent.__new__(TariffPricingAgent)
            pricing._theta = np.array([1.2, 4.0, 4.0])
            pricing._max_retries = 1
            pricing._model = None
            orch.pricing_agent = pricing

            from unittest.mock import MagicMock
            monitor = MagicMock()
            monitor.summary.return_value = pd.DataFrame()
            orch.monitor_agent = monitor

            orch.export_predictions(out_path)

        assert Path(out_path).exists()
        df = pd.read_csv(out_path)
        expected_cols = {
            "hourly_timestamp", "hour_of_day", "is_weekend",
            "actual_urban_mean_utilization", "actual_urban_peak_queue",
            "pred_urban_mean_utilization", "pred_urban_peak_queue",
        }
        assert expected_cols.issubset(set(df.columns)), \
            f"Missing columns: {expected_cols - set(df.columns)}"
