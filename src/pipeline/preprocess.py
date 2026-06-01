# =============================================================================
# src/pipeline/preprocess.py — OP'26 Data Preprocessing Pipeline
#
# Ingests ACN-Data (Caltech/JPL) and UrbanEV ST-EVCDP datasets,
# engineers features, and produces the unified analytical base CSV.
#
# Assumptions (documented inline):
#   • Zero/null kWh sessions are excluded — they represent incomplete charges
#     and would corrupt hourly revenue aggregates.
#   • Charger Utilisation Rate = avg_duration / 60, clipped to [0,1].
#     avg_duration is in minutes; each time slot represents 60 minutes of
#     available time. This matches the PS definition: Charging Time / Total
#     Available Time.
#   • Queue length proxy = floor(volume × (1 − utilisation) × 0.4).
#     This approximates vehicles waiting when chargers are near capacity.
#   • ACN and UrbanEV are aligned by positional index (not by wall-clock time)
#     because the two datasets cover different geographies and time periods.
#
# Limitations:
#   • Positional alignment means ACN timestamps are used as the master index
#     but UrbanEV rows do not correspond to the same real-world hours.
#     Temporal patterns from UrbanEV are treated as representative demand
#     profiles, not as co-located observations.
#   • The 0.4 queue scaling factor is a heuristic; no ground-truth queue
#     data is available for calibration.
#   • Revenue figures are simulated at ₹15/kWh baseline; actual tariffs
#     and currency conversions are not applied.
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import P_BASE, RAW_ACN_PATH, RAW_URBAN_DIR, PROCESSED_BASE_PATH

logger = logging.getLogger("ev_agentic.pipeline")

# Required UrbanEV CSV files
_URBAN_FILES = ["volume.csv", "occupancy.csv", "duration.csv"]


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_acn(path: str) -> pd.DataFrame:
    """
    Load ACN Excel workbook, parse timestamps, and exclude zero/null kWh rows.

    Assumption: zero/null kWhDelivered rows represent incomplete or aborted
    sessions and are excluded before aggregation to avoid corrupting revenue
    and energy totals.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"ACN data file not found: '{path}'\n"
            "Required by _load_acn(). Place acndata_sessions.json.xlsx in data/raw/."
        )

    df = pd.read_excel(path)
    logger.info("ACN raw rows loaded: %d", len(df))

    # Detect column names flexibly
    col_map: dict[str, str] = {}
    for col in df.columns:
        cl = col.lower()
        if "connect" in cl and "dis" not in cl:
            col_map["connectionTime"] = col
        elif "disconnect" in cl:
            col_map["disconnectTime"] = col
        elif "kwh" in cl or "energy" in cl:
            col_map["kWhDelivered"] = col
        elif "station" in cl:
            col_map["stationID"] = col

    required = ["connectionTime", "kWhDelivered", "stationID"]
    missing = [k for k in required if k not in col_map]
    if missing:
        raise ValueError(
            f"ACN dataset missing expected columns: {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )

    # Parse timestamps
    df["connect_dt"] = pd.to_datetime(df[col_map["connectionTime"]], errors="coerce")
    df.dropna(subset=["connect_dt"], inplace=True)

    # Assumption: exclude zero/null kWh rows
    kwh_col = col_map["kWhDelivered"]
    zero_null_mask = df[kwh_col].isna() | (df[kwh_col] <= 0)
    excluded = zero_null_mask.sum()
    if excluded > 0:
        logger.warning(
            "ACN: excluding %d zero/null kWh rows (%.1f%% of loaded rows)",
            excluded, excluded / len(df) * 100,
        )
    df = df[~zero_null_mask].copy()

    df["kWhDelivered"] = df[kwh_col].astype(float)
    df["stationID"] = df[col_map["stationID"]].astype(str)
    # Round connection time to nearest hour for aggregation
    df["hourly_timestamp"] = df["connect_dt"].dt.round("h")
    # Baseline revenue per session at ₹15/kWh
    df["baseline_revenue"] = df["kWhDelivered"] * P_BASE

    logger.info("ACN rows after zero/null kWh exclusion: %d", len(df))
    return df


def _load_urban(urban_dir: str) -> pd.DataFrame:
    """
    Load UrbanEV wide-format matrices (volume, occupancy, duration),
    melt to long format, merge on [time_step, station_node], and validate nulls.

    Assumption: occupancy_density × 1.2 clipped to [0,1] gives charger
    utilisation rate. Queue proxy = floor(volume × (1 − util) × 0.4).
    """
    base = Path(urban_dir)
    for fname in _URBAN_FILES:
        fpath = base / fname
        if not fpath.exists():
            raise FileNotFoundError(
                f"UrbanEV file not found: '{fpath}'\n"
                f"Required by _load_urban(). Place {fname} in {urban_dir}/."
            )

    def _melt(fname: str, value_name: str) -> pd.DataFrame:
        df = pd.read_csv(base / fname)
        index_col = df.columns[0]
        melted = df.melt(id_vars=[index_col], var_name="station_node", value_name=value_name)
        melted.rename(columns={index_col: "time_step"}, inplace=True)
        return melted

    vol_long = _melt("volume.csv", "traffic_volume")
    occ_long = _melt("occupancy.csv", "occupancy_density")
    dur_long = _melt("duration.csv", "avg_duration")

    # Merge on [time_step, station_node]
    df = vol_long.merge(occ_long, on=["time_step", "station_node"])
    df = df.merge(dur_long, on=["time_step", "station_node"])

    # Validate no unexpected nulls in key columns
    key_cols = ["traffic_volume", "occupancy_density", "avg_duration"]
    null_counts = df[key_cols].isna().sum()
    if null_counts.any():
        raise ValueError(
            f"Unexpected nulls after UrbanEV merge: {null_counts[null_counts > 0].to_dict()}"
        )

    # Charger Utilization Rate = Charging Time / Total Available Time
    # avg_duration is in minutes; we assume each time_step represents 60 minutes
    # of available time per station. Clipped to [0, 1].
    # Assumption: avg_duration is in minutes; 60 min = 1 hour per time slot.
    df["charger_utilization"] = (df["avg_duration"] / 60.0).clip(0.0, 1.0)
    # Queue Length Proxy: vehicles waiting when chargers are near capacity
    # Formula: floor(volume × (1 − utilisation) × 0.4)
    df["queue_length_proxy"] = (
        df["traffic_volume"] * (1.0 - df["charger_utilization"]) * 0.4
    ).apply(np.floor)

    logger.info("UrbanEV long-format rows: %d", len(df))
    return df


def _aggregate_acn_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate ACN sessions to hourly granularity."""
    agg = df.groupby("hourly_timestamp").agg(
        acn_sessions_count=("stationID", "count"),
        acn_total_kwh=("kWhDelivered", "sum"),
        acn_base_revenue=("baseline_revenue", "sum"),
        acn_avg_kwh_per_session=("kWhDelivered", "mean"),
    ).reset_index()

    # Revenue per Session = total revenue / session count
    agg["acn_revenue_per_session"] = (
        agg["acn_base_revenue"] / agg["acn_sessions_count"].clip(lower=1)
    )
    # Energy Cost per kWh = baseline (₹15/kWh fixed)
    agg["acn_energy_cost_per_kwh"] = P_BASE

    if len(agg) == 0:
        raise ValueError("ACN hourly aggregation produced zero rows. Check input data.")

    logger.info("ACN hourly rows: %d", len(agg))
    return agg


def _aggregate_urban_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate UrbanEV spatial data to hourly (time_step) granularity."""
    agg = df.groupby("time_step").agg(
        urban_mean_utilization=("charger_utilization", "mean"),
        urban_peak_queue=("queue_length_proxy", "max"),
        urban_total_volume=("traffic_volume", "sum"),
    ).reset_index()

    if len(agg) == 0:
        raise ValueError("UrbanEV hourly aggregation produced zero rows. Check input data.")

    logger.info("UrbanEV hourly rows: %d", len(agg))
    return agg


def _align_and_merge(
    acn_hourly: pd.DataFrame,
    urban_hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Align ACN and UrbanEV hourly aggregates by positional index.

    The two datasets cover different geographies and time periods, so
    alignment is positional rather than by wall-clock timestamp. The ACN
    hourly_timestamp is used as the master time index.
    """
    logger.info(
        "Aligning: ACN=%d rows, UrbanEV=%d rows",
        len(acn_hourly), len(urban_hourly),
    )
    min_len = min(len(acn_hourly), len(urban_hourly))

    merged = pd.concat(
        [
            acn_hourly.iloc[:min_len].reset_index(drop=True),
            urban_hourly.iloc[:min_len].reset_index(drop=True),
        ],
        axis=1,
    )

    # Temporal features derived from ACN timestamp
    merged["hour_of_day"] = merged["hourly_timestamp"].dt.hour
    merged["day_of_week"] = merged["hourly_timestamp"].dt.dayofweek
    merged["is_weekend"] = (merged["day_of_week"] >= 5).astype(int)

    logger.info("Unified analytical base rows: %d", len(merged))
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    acn_path: str = RAW_ACN_PATH,
    urban_dir: str = RAW_URBAN_DIR,
    output_path: str = PROCESSED_BASE_PATH,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline. Callable as a function (not only __main__).

    Steps:
      1. Load and validate ACN sessions (zero/null kWh excluded)
      2. Load and validate UrbanEV spatial matrices
      3. Aggregate both to hourly granularity
      4. Align by positional index and merge
      5. Write unified_analytical_base.csv to output_path

    Returns the unified DataFrame.
    Raises FileNotFoundError if any required input file is missing.
    """
    logger.info("=== OP'26 Preprocessing Pipeline START ===")

    acn_raw = _load_acn(acn_path)
    urban_raw = _load_urban(urban_dir)

    acn_hourly = _aggregate_acn_hourly(acn_raw)
    urban_hourly = _aggregate_urban_hourly(urban_raw)

    unified = _align_and_merge(acn_hourly, urban_hourly)

    # Write output
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    unified.to_csv(out, index=False)
    logger.info("Unified base written → %s  (%d rows)", output_path, len(unified))
    logger.info("=== OP'26 Preprocessing Pipeline COMPLETE ===")

    return unified


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from src.utils.logging_utils import configure_logging, log_dependency_versions

    parser = argparse.ArgumentParser(description="OP'26 Preprocessing Pipeline")
    parser.add_argument("--acn", default=RAW_ACN_PATH, help="Path to ACN Excel file")
    parser.add_argument("--urban-dir", default=RAW_URBAN_DIR, help="Directory with UrbanEV CSVs")
    parser.add_argument("--out", default=PROCESSED_BASE_PATH, help="Output CSV path")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    log_dependency_versions()
    run_pipeline(acn_path=args.acn, urban_dir=args.urban_dir, output_path=args.out)
