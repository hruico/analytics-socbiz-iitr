"""Demand prediction agent using XGBoost."""
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from typing import Tuple, Dict


class DemandAgent:
    """Predicts utilization, queue, and congestion probability."""
    
    FEATURES = [
        'acn_sessions_count', 'acn_total_kwh', 'acn_avg_kwh_per_session',
        'hour_of_day', 'day_of_week', 'is_weekend', 'is_peak_hour'
    ]
    
    TARGETS = ['urban_mean_utilization', 'urban_peak_queue']
    
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.model = MultiOutputRegressor(
            XGBRegressor(
                n_estimators=100,  # Reduced for prototype
                learning_rate=0.04,
                max_depth=6,
                subsample=0.80,
                colsample_bytree=0.75,
                random_state=random_seed,
                tree_method="hist"
            )
        )
        self.is_trained = False
    
    def train(self, df: pd.DataFrame) -> Dict[str, float]:
        """Train the demand model."""
        X = df[self.FEATURES].values
        y = df[self.TARGETS].values
        
        self.model.fit(X, y)
        self.is_trained = True
        
        return {"status": "trained", "samples": len(df)}
    
    def predict(self, features: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict utilization, queue, and congestion probability.
        
        Returns:
            (u_pred, q_pred, congestion_prob)
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X = features[self.FEATURES].values
        preds = self.model.predict(X)
        
        # Clip predictions
        u_pred = np.clip(preds[:, 0], 0, 1)
        q_pred = np.clip(preds[:, 1], 0, np.inf)
        
        # Simple congestion probability: high queue = high congestion
        congestion_prob = np.clip(q_pred / 10.0, 0, 1)
        
        return u_pred, q_pred, congestion_prob
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict[str, float]:
        """Evaluate model on test set."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        X = X_test[self.FEATURES].values
        y_true = y_test[self.TARGETS].values
        preds = self.model.predict(X)
        
        # Simple RMSE
        rmse_util = np.sqrt(np.mean((preds[:, 0] - y_true[:, 0]) ** 2))
        rmse_queue = np.sqrt(np.mean((preds[:, 1] - y_true[:, 1]) ** 2))
        
        return {
            "rmse_utilization": rmse_util,
            "rmse_queue": rmse_queue
        }
