"""Demand prediction agent using XGBoost."""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from typing import Tuple, Dict
import logging

logger = logging.getLogger(__name__)

ACN_AVG_KWH = 9.0


class DemandAgent:
    """Predicts charger utilization, queue length, and congestion probability."""

    ALL_FEATURES = [
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour',
        'urban_mean_utilization',
        'total_volume', 'rolling_3h_volume', 'queue_length_proxy',
        'occupancy_density', 'count', 'fast_count', 'CBD', 'dynamic_pricing',
    ]

    MINIMAL_FEATURES = [
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour',
        'urban_mean_utilization',
    ]

    TARGETS = ['urban_mean_utilization']

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.80,
                colsample_bytree=0.80,
                random_state=random_seed,
                tree_method="hist"
            )
        )
        self.is_trained = False
        self.features_used: list = []

    def _resolve_features(self, df: pd.DataFrame) -> list:
        """Use richest available feature set."""
        available = [f for f in self.ALL_FEATURES if f in df.columns]
        if len(available) >= len(self.MINIMAL_FEATURES):
            return available
        return [f for f in self.MINIMAL_FEATURES if f in df.columns]

    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train on available features, auto-detected from dataframe columns."""
        self.features_used = self._resolve_features(df)
        X = df[self.features_used].values
        y = df[self.TARGETS].values
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"DemandAgent trained on {len(self.features_used)} features: {self.features_used}")
        return {"status": "trained", "samples": len(df), "features": len(self.features_used)}

    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (u_pred, q_pred, congestion_prob).
        Queue uses real proxy column when available, falls back to utilization heuristic.
        """
        if not self.is_trained:
            raise ValueError("Model not trained")

        X = features[self.features_used].values
        preds = self.model.predict(X)
        u_pred = np.clip(preds[:, 0] if preds.ndim > 1 else preds, 0, 1)

        if 'queue_length_proxy' in features.columns:
            q_pred = features['queue_length_proxy'].values.astype(float)
        else:
            q_pred = np.maximum(0, 10 * (u_pred - 0.5))

        congestion_prob = np.clip(u_pred, 0, 1)
        return u_pred, q_pred, congestion_prob

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        """Compute RMSE on held-out test set."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        X = X_test[self.features_used].values
        y_true = y_test[self.TARGETS].values
        preds = self.model.predict(X)
        rmse_util = np.sqrt(np.mean((preds[:, 0] - y_true[:, 0]) ** 2))
        return {"rmse_utilization": rmse_util}
