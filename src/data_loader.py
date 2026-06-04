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


def train_test_split(df: pd.DataFrame, train_ratio: float = 0.70, stratify_by_regime: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset chronologically or with regime stratification.
    
    Args:
        df: Input dataframe sorted by time_step
        train_ratio: Fraction for training set
        stratify_by_regime: If True, ensure balanced regime representation in test set
        
    Returns:
        (train_df, test_df) tuple
    """
    if stratify_by_regime and 'urban_mean_utilization' in df.columns:
        # Classify regimes
        df = df.copy()
        df['_regime'] = 'neutral'
        df.loc[df['urban_mean_utilization'] > 0.80, '_regime'] = 'surge'
        df.loc[df['urban_mean_utilization'] < 0.30, '_regime'] = 'discount'
        
        # Sample from each regime proportionally
        from sklearn.model_selection import train_test_split as sklearn_split
        
        train_df, test_df = sklearn_split(
            df, 
            train_size=train_ratio,
            stratify=df['_regime'],
            random_state=42
        )
        
        # Remove temp column and sort by time_step
        train_df = train_df.drop('_regime', axis=1).sort_values('time_step').reset_index(drop=True)
        test_df = test_df.drop('_regime', axis=1).sort_values('time_step').reset_index(drop=True)
        
        # Log regime distribution
        import logging
        logger = logging.getLogger(__name__)
        test_regimes = []
        for _, row in test_df.iterrows():
            if row['urban_mean_utilization'] > 0.80:
                test_regimes.append('surge')
            elif row['urban_mean_utilization'] < 0.30:
                test_regimes.append('discount')
            else:
                test_regimes.append('neutral')
        
        from collections import Counter
        regime_counts = Counter(test_regimes)
        logger.info(f"Stratified test set regime distribution:")
        for regime, count in regime_counts.items():
            logger.info(f"  {regime}: {count} rows ({count/len(test_df)*100:.1f}%)")
    else:
        # Chronological split
        split_idx = int(len(df) * train_ratio)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
    
    if len(test_df) < 10:
        raise ValueError(f"Test set too small: {len(test_df)} rows. Need at least 10.")
    
    return train_df, test_df
