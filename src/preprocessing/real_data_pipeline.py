"""Builds the unified analytical base from raw ACN and UrbanEV datasets."""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class RealDataPipeline:
    """Processes raw ACN sessions and UrbanEV grid data into a single analytical base."""

    def __init__(self, baseline_tariff: float = 15.0):
        self.baseline_tariff = baseline_tariff

    def load_acn_data(self, sessions_path: str) -> pd.DataFrame:
        """Parse ACN session JSON and extract temporal features."""
        with open(sessions_path, "r") as f:
            data = json.load(f)
        sessions = data if isinstance(data, list) else data.get("sessions", [])
        if not sessions:
            return pd.DataFrame()

        records = [
            {
                "sessionID":      s["sessionID"],
                "stationID":      s.get("stationID", "unknown"),
                "connectionTime": pd.to_datetime(s["connectionTime"], utc=True),
                "disconnectTime": pd.to_datetime(s["disconnectTime"], utc=True),
                "kWhDelivered":   float(s["kWhDelivered"]),
            }
            for s in sessions
        ]
        df = pd.DataFrame(records)
        df["hour_of_day"] = df["connectionTime"].dt.hour
        df["day_of_week"] = df["connectionTime"].dt.dayofweek
        df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
        return df

    def compute_peak_hours(self, acn_df: pd.DataFrame, percentile: float = 0.75) -> set:
        """Identify peak hours from actual session counts (data-driven, not hardcoded)."""
        hourly_counts = acn_df.groupby("hour_of_day")["sessionID"].count()
        threshold = hourly_counts.quantile(percentile)
        peak_hours = set(hourly_counts[hourly_counts >= threshold].index)
        logger.info(f"Peak hours (>P{int(percentile*100)}): {sorted(peak_hours)}")
        return peak_hours

    def aggregate_acn_hourly(self, acn_df: pd.DataFrame, peak_hours: set) -> pd.DataFrame:
        """Aggregate ACN sessions to unique (hour, day_of_week, is_weekend) patterns."""
        if acn_df.empty:
            return pd.DataFrame()

        acn_hourly = (
            acn_df.groupby(["hour_of_day", "day_of_week", "is_weekend"])
            .agg(
                acn_sessions_count=("sessionID", "count"),
                acn_total_kwh=("kWhDelivered", "sum"),
                acn_avg_kwh_per_session=("kWhDelivered", "mean"),
            )
            .reset_index()
        )
        acn_hourly["is_peak_hour"]    = acn_hourly["hour_of_day"].isin(peak_hours).astype(int)
        acn_hourly["acn_base_revenue"] = acn_hourly["acn_total_kwh"] * self.baseline_tariff
        return acn_hourly

    def load_urbanev_data(self, data_dir: str):
        """Load UrbanEV raw files: occupancy, volume, time index, station info."""
        return (
            pd.read_csv(f"{data_dir}/occupancy.csv"),
            pd.read_csv(f"{data_dir}/volume.csv"),
            pd.read_csv(f"{data_dir}/time.csv"),
            pd.read_csv(f"{data_dir}/information.csv"),
        )

    def compute_utilization(self, occupancy: pd.DataFrame, info_df: pd.DataFrame) -> pd.DataFrame:
        """
        Per-zone utilization = occupancy / charger_count per zone.

        Uses a blended metric (40% mean + 60% P90 across zones per timestep)
        to expose high-demand grids without being dominated by single outliers.
        The naive system-wide mean collapses to 22-34% with no surge signal.
        """
        zone_capacity   = info_df.set_index("grid")["count"]
        timestamp_col   = occupancy["timestamp"]
        occ_matrix      = occupancy.drop("timestamp", axis=1)
        occ_matrix.columns = occ_matrix.columns.astype(int)

        util_df   = occ_matrix.div(zone_capacity, axis=1)
        mean_util = util_df.mean(axis=1)
        p90_util  = util_df.quantile(0.90, axis=1)
        utilization = 0.4 * mean_util + 0.6 * p90_util

        logger.info(f"Utilization (40% mean + 60% P90): {utilization.min():.2%} – {utilization.max():.2%}")
        return pd.DataFrame({"timestamp": timestamp_col, "urban_mean_utilization": utilization})

    def aggregate_urbanev_hourly(
        self,
        utilization_df: pd.DataFrame,
        time_df: pd.DataFrame,
        volume_df: pd.DataFrame = None,
        occupancy_raw: pd.DataFrame = None,
        info_df: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """
        Resample UrbanEV 5-min data to hourly temporal patterns.
        Merges utilization with volume, queue proxy, occupancy density,
        spatial flags (CBD, dynamic_pricing), and rolling demand features.
        """
        time_df = time_df.copy()
        time_df["datetime"]  = pd.to_datetime(time_df[["year", "month", "day", "hour", "minute", "second"]])
        time_df["timestamp"] = range(1, len(time_df) + 1)

        merged = utilization_df.merge(time_df[["timestamp", "datetime"]], on="timestamp", how="left")
        merged["hour_of_day"] = merged["datetime"].dt.hour
        merged["day_of_week"] = merged["datetime"].dt.dayofweek
        merged["is_weekend"]  = (merged["day_of_week"] >= 5).astype(int)

        # Volume features
        vol_hourly = None
        if volume_df is not None:
            v = volume_df.copy()
            v.rename(columns={v.columns[0]: "timestamp"}, inplace=True)
            v = v.melt(id_vars="timestamp", var_name="gridID", value_name="volume")
            v["timestamp"] = v["timestamp"].astype(int)
            v["gridID"]    = v["gridID"].astype(int)
            v = v.merge(time_df[["timestamp", "datetime"]], on="timestamp", how="left")
            v["hour_of_day"] = v["datetime"].dt.hour
            v["day_of_week"] = v["datetime"].dt.dayofweek
            v["is_weekend"]  = (v["day_of_week"] >= 5).astype(int)
            v = v.sort_values(["gridID", "datetime"])
            v["rolling_3h_volume"] = (
                v.groupby("gridID")["volume"]
                .transform(lambda x: x.rolling(window=36, min_periods=1).mean())
            )
            vol_hourly = v.groupby(["hour_of_day", "day_of_week", "is_weekend"]).agg(
                total_volume=("volume", "sum"),
                rolling_3h_volume=("rolling_3h_volume", "mean"),
            ).reset_index()

        # Occupancy-derived features
        occ_hourly = None
        if occupancy_raw is not None and info_df is not None:
            o = occupancy_raw.copy()
            o.rename(columns={o.columns[0]: "timestamp"}, inplace=True)
            o = o.melt(id_vars="timestamp", var_name="gridID", value_name="occupancy")
            o["timestamp"] = o["timestamp"].astype(int)
            o["gridID"]    = o["gridID"].astype(int)

            info_sub = info_df[["grid", "count", "fast_count", "area", "CBD", "dynamic_pricing"]].copy()
            info_sub.rename(columns={"grid": "gridID"}, inplace=True)
            o = o.merge(info_sub, on="gridID", how="left")

            o["queue_length_proxy"] = (
                (o["occupancy"] / o["count"].replace(0, np.nan)) - 1
            ).clip(lower=0).fillna(0)
            o["occupancy_density"] = (o["occupancy"] / o["area"].replace(0, np.nan)).fillna(0)

            o = o.merge(time_df[["timestamp", "datetime"]], on="timestamp", how="left")
            o["hour_of_day"] = o["datetime"].dt.hour
            o["day_of_week"] = o["datetime"].dt.dayofweek
            o["is_weekend"]  = (o["day_of_week"] >= 5).astype(int)

            occ_hourly = o.groupby(["hour_of_day", "day_of_week", "is_weekend"]).agg(
                queue_length_proxy=("queue_length_proxy", "mean"),
                occupancy_density=("occupancy_density", "mean"),
                count=("count", "mean"),
                fast_count=("fast_count", "mean"),
                CBD=("CBD", "first"),
                dynamic_pricing=("dynamic_pricing", "first"),
            ).reset_index()

        # Core utilization hourly
        merged_idx = merged.set_index("datetime")
        urban_hourly = merged_idx.resample("1h").agg(
            urban_mean_utilization=("urban_mean_utilization", "mean"),
            hour_of_day=("hour_of_day", "first"),
            day_of_week=("day_of_week", "first"),
            is_weekend=("is_weekend", "first"),
        ).reset_index(drop=True)
        urban_hourly = (
            urban_hourly.groupby(["hour_of_day", "day_of_week", "is_weekend"])["urban_mean_utilization"]
            .mean()
            .reset_index()
        )

        join_keys = ["hour_of_day", "day_of_week", "is_weekend"]
        if vol_hourly is not None:
            urban_hourly = urban_hourly.merge(vol_hourly, on=join_keys, how="left")
        if occ_hourly is not None:
            urban_hourly = urban_hourly.merge(occ_hourly, on=join_keys, how="left")

        for col in ["total_volume", "rolling_3h_volume", "queue_length_proxy", "occupancy_density", "count", "fast_count"]:
            if col in urban_hourly.columns:
                urban_hourly[col] = urban_hourly[col].fillna(0)
        for col in ["CBD", "dynamic_pricing"]:
            if col in urban_hourly.columns:
                urban_hourly[col] = urban_hourly[col].fillna(0)

        return urban_hourly

    def build_unified_analytical_base(
        self,
        acn_sessions_path: str,
        urbanev_data_dir: str,
        output_path: str = "data/processed/unified_analytical_base.csv",
    ) -> pd.DataFrame:
        """
        Join ACN hourly patterns with UrbanEV hourly patterns on
        (hour_of_day, day_of_week, is_weekend) — behavioural alignment,
        not calendar alignment (datasets span different years/geographies).
        """
        logger.info("Building unified analytical base...")

        acn_df    = self.load_acn_data(acn_sessions_path)
        logger.info(f"ACN: {len(acn_df)} sessions")
        peak_hours = self.compute_peak_hours(acn_df)
        acn_hourly = self.aggregate_acn_hourly(acn_df, peak_hours)

        occupancy, volume_df, time_df, info_df = self.load_urbanev_data(urbanev_data_dir)
        utilization_df = self.compute_utilization(occupancy, info_df)
        urban_hourly   = self.aggregate_urbanev_hourly(
            utilization_df, time_df,
            volume_df=volume_df, occupancy_raw=occupancy, info_df=info_df,
        )
        logger.info(f"UrbanEV hourly: {len(urban_hourly)} temporal patterns")

        unified = acn_hourly.merge(urban_hourly, on=["hour_of_day", "day_of_week", "is_weekend"], how="inner")
        logger.info(f"Unified: {len(unified)} rows | util range: {unified['urban_mean_utilization'].min():.2%} – {unified['urban_mean_utilization'].max():.2%}")

        unified = unified.sort_values(["hour_of_day", "day_of_week"]).reset_index(drop=True)
        unified["time_step"] = range(len(unified))

        core = ["time_step", "hour_of_day", "day_of_week", "is_weekend", "is_peak_hour",
                "acn_sessions_count", "acn_total_kwh", "acn_avg_kwh_per_session", "acn_base_revenue",
                "urban_mean_utilization"]
        extra = ["total_volume", "rolling_3h_volume", "queue_length_proxy",
                 "occupancy_density", "count", "fast_count", "CBD", "dynamic_pricing"]
        unified = unified[core + [c for c in extra if c in unified.columns]]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        unified.to_csv(output_path, index=False)
        logger.info(f"✓ Exported to {output_path}")
        return unified


def build_real_data_pipeline() -> pd.DataFrame:
    return RealDataPipeline(baseline_tariff=15.0).build_unified_analytical_base(
        acn_sessions_path="data/raw/acndata_sessions.json",
        urbanev_data_dir="data/raw",
        output_path="data/processed/unified_analytical_base.csv",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    build_real_data_pipeline()
