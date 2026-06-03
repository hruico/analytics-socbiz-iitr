"""Minimal data loader for unified analytical base."""
import pandas as pd
import numpy as np
from typing import Tuple


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load unified analytical base CSV.
    
    FIX 4: is_peak_hour now computed from data in preprocessing, not hardcoded here.
    """
    df = pd.read_csv(path)
    
    # is_peak_hour should already exist from real_data_pipeline
    if 'is_peak_hour' not in df.columns:
        # Fallback only if missing (shouldn't happen with new pipeline)
        df['is_peak_hour'] = 0
    
    # Engineer revenue_per_kwh
    df['revenue_per_kwh'] = np.where(
        df['acn_total_kwh'] > 0,
        df['acn_base_revenue'] / df['acn_total_kwh'],
        0
    )
    
    # Handle missing values
    if 'urban_mean_utilization' in df.columns:
        df['urban_mean_utilization'] = df['urban_mean_utilization'].fillna(method='ffill')
    if 'urban_peak_queue' in df.columns:
        df['urban_peak_queue'] = df['urban_peak_queue'].fillna(method='ffill')
    
    # Sort by time_step
    df = df.sort_values('time_step').reset_index(drop=True)
    
    return df


def train_test_split(df: pd.DataFrame, train_ratio: float = 0.80) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset chronologically.
    
    Args:
        df: Input dataframe sorted by time_step
        train_ratio: Fraction for training set
        
    Returns:
        (train_df, test_df) tuple
    """
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    if len(test_df) < 10:
        raise ValueError(f"Test set too small: {len(test_df)} rows. Need at least 10.")
    
    return train_df, test_df
