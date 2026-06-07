"""Data loading and train/test split utilities."""
import pandas as pd
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def load_dataset(path: str) -> pd.DataFrame:
    """Load unified analytical base and add derived columns."""
    df = pd.read_csv(path)

    if "is_peak_hour" not in df.columns:
        df["is_peak_hour"] = 0

    df["revenue_per_kwh"] = np.where(
        df["acn_total_kwh"] > 0,
        df["acn_base_revenue"] / df["acn_total_kwh"],
        0,
    )

    if "urban_mean_utilization" in df.columns:
        df["urban_mean_utilization"] = df["urban_mean_utilization"].ffill()
    if "urban_peak_queue" in df.columns:
        df["urban_peak_queue"] = df["urban_peak_queue"].ffill()

    return df.sort_values("time_step").reset_index(drop=True)


def train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    stratify_by_regime: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset with optional regime stratification.

    Stratification ensures surge, neutral, and discount rows all appear in
    both train and test sets, preventing a test set dominated by one regime.
    Falls back to chronological split when stratification is not needed.
    """
    if stratify_by_regime and "urban_mean_utilization" in df.columns:
        df = df.copy()
        df["_regime"] = "neutral"
        df.loc[df["urban_mean_utilization"] > 0.80, "_regime"] = "surge"
        df.loc[df["urban_mean_utilization"] < 0.30, "_regime"] = "discount"

        from sklearn.model_selection import train_test_split as sklearn_split
        from collections import Counter

        train_df, test_df = sklearn_split(
            df, train_size=train_ratio, stratify=df["_regime"], random_state=42
        )
        train_df = train_df.drop("_regime", axis=1).sort_values("time_step").reset_index(drop=True)
        test_df  = test_df.drop("_regime", axis=1).sort_values("time_step").reset_index(drop=True)

        counts = Counter(
            "surge" if r > 0.80 else "discount" if r < 0.30 else "neutral"
            for r in test_df["urban_mean_utilization"]
        )
        logger.info(f"Test set regime distribution: {dict(counts)}")
    else:
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df  = df.iloc[split_idx:].copy()

    if len(test_df) < 10:
        raise ValueError(f"Test set too small: {len(test_df)} rows.")

    return train_df, test_df
