# =============================================================================
# src/agents/demand.py — DemandPredictionAgent
#
# Loads the unified analytical base CSV, engineers causal features, applies a
# strict chronological 80/20 train/test split, and trains a MultiOutputRegressor
# wrapping XGBRegressor (or optionally LightGBM) to jointly predict:
#   • urban_mean_utilization
#   • urban_peak_queue
# =============================================================================

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.config import (
    PROCESSED_BASE_PATH,
    TRAIN_RATIO,
    XGB_PARAMS,
    LGB_PARAMS,
    RANDOM_STATE,
    FEATURE_COLS,
    engineer_features,
    ForecastState,
)

logger = logging.getLogger(__name__)

# Prediction targets
_TARGETS: list[str] = ["urban_mean_utilization", "urban_peak_queue"]


class DemandPredictionAgent:
    """
    XGBoost-based multi-output demand forecaster.

    Trains on a strict chronological 80/20 split of the unified analytical base
    so that no future data leaks into the training set.

    Parameters
    ----------
    csv_path : str
        Path to the unified analytical base CSV produced by the pipeline.
    use_lightgbm : bool
        When True, also trains a LightGBM backend alongside XGBoost.
        Enables ``compare_backends()`` and LightGBM-based predictions.
    """

    def __init__(
        self,
        csv_path: str = PROCESSED_BASE_PATH,
        use_lightgbm: bool = False,
    ) -> None:
        # ── 1. Load CSV ──────────────────────────────────────────────────────
        logger.info("DemandPredictionAgent: loading data from '%s'", csv_path)
        df_raw = pd.read_csv(csv_path)

        # ── 2. Feature engineering (single source of truth in src/config.py) ─
        df = engineer_features(df_raw)

        # ── 3. Strict chronological 80/20 split — NO shuffling ───────────────
        n_total = len(df)
        n_train = int(n_total * TRAIN_RATIO)

        self.train_df: pd.DataFrame = df.iloc[:n_train].reset_index(drop=True)
        self.test_df: pd.DataFrame = df.iloc[n_train:].reset_index(drop=True)

        self.feature_cols: list[str] = FEATURE_COLS
        self.targets: list[str] = _TARGETS

        logger.info(
            "DemandPredictionAgent: train rows=%d  test rows=%d",
            len(self.train_df),
            len(self.test_df),
        )
        logger.info("DemandPredictionAgent: features=%s", self.feature_cols)

        # ── 4. Train XGBoost MultiOutputRegressor ────────────────────────────
        X_train = self.train_df[self.feature_cols].values
        y_train = self.train_df[self.targets].values

        self.model: MultiOutputRegressor = MultiOutputRegressor(
            XGBRegressor(**XGB_PARAMS, random_state=RANDOM_STATE)
        )
        logger.info("DemandPredictionAgent: training XGBoost model …")
        self.model.fit(X_train, y_train)
        logger.info("DemandPredictionAgent: XGBoost training complete.")

        # ── 5. Optional LightGBM backend ─────────────────────────────────────
        self._use_lightgbm: bool = use_lightgbm
        self._lgb_model: Optional[MultiOutputRegressor] = None

        if use_lightgbm:
            try:
                from lightgbm import LGBMRegressor  # type: ignore

                self._lgb_model = MultiOutputRegressor(
                    LGBMRegressor(**LGB_PARAMS, random_state=RANDOM_STATE)
                )
                logger.info("DemandPredictionAgent: training LightGBM model …")
                self._lgb_model.fit(X_train, y_train)
                logger.info("DemandPredictionAgent: LightGBM training complete.")
            except ImportError:
                logger.warning(
                    "DemandPredictionAgent: LightGBM not installed — "
                    "falling back to XGBoost only."
                )
                self._use_lightgbm = False

    # ── Public interface ──────────────────────────────────────────────────────

    def predict_state(self, row_idx: int) -> ForecastState:
        """
        Return a validated ``ForecastState`` for the test-set row at *row_idx*.

        ``u_pred`` is clipped to [0, 1] (enforced by the Pydantic validator).
        ``kwh_delivered`` is floored at 0.01.

        Requirements: 4.4
        """
        # Extract feature row and run prediction
        X = self.test_df[self.feature_cols].iloc[[row_idx]].values
        preds = self.model.predict(X)  # shape (1, 2)
        u_pred = preds[0, 0]
        q_pred = max(0.0, preds[0, 1])  # floor at 0 — XGBoost can predict tiny negatives

        # Actual target values
        u_actual = self.test_df["urban_mean_utilization"].iloc[row_idx]
        q_actual = max(0.0, float(self.test_df["urban_peak_queue"].iloc[row_idx]))

        # kWh delivered — floored at 0.01
        kwh = self.test_df["acn_total_kwh"].iloc[row_idx]
        kwh_delivered = max(0.01, float(kwh))

        # Contextual fields
        hour_of_day = int(self.test_df["hour_of_day"].iloc[row_idx])
        is_weekend = int(self.test_df["is_weekend"].iloc[row_idx])
        timestamp = str(self.test_df["hourly_timestamp"].iloc[row_idx])

        return ForecastState(
            timestamp=timestamp,
            u_pred=float(u_pred),       # Pydantic validator clips to [0, 1]
            q_pred=float(q_pred),       # floored at 0 above
            u_actual=float(u_actual),   # Pydantic validator clips to [0, 1]
            q_actual=float(q_actual),   # floored at 0 above
            kwh_delivered=kwh_delivered,
            hour_of_day=hour_of_day,
            is_weekend=is_weekend,
        )

    def evaluation_metrics(self) -> dict[str, dict[str, float]]:
        """
        Return RMSE, MAE, and R² for both prediction targets evaluated on the
        held-out test set.

        Returns
        -------
        dict
            ``{target: {"RMSE": float, "MAE": float, "R2": float}}``
            for each of ``urban_mean_utilization`` and ``urban_peak_queue``.

        Requirements: 4.3
        """
        X_test = self.test_df[self.feature_cols].values
        y_test = self.test_df[self.targets].values
        y_pred = self.model.predict(X_test)

        metrics: dict[str, dict[str, float]] = {}
        for i, target in enumerate(self.targets):
            y_t = y_test[:, i]
            y_p = y_pred[:, i]
            metrics[target] = {
                "RMSE": float(np.sqrt(mean_squared_error(y_t, y_p))),
                "MAE": float(mean_absolute_error(y_t, y_p)),
                "R2": float(r2_score(y_t, y_p)),
            }
        return metrics

    def compare_backends(self) -> pd.DataFrame:
        """
        Return a side-by-side RMSE/MAE/R² comparison of XGBoost and LightGBM.

        Only available when ``use_lightgbm=True`` and LightGBM was successfully
        installed and trained.

        Returns
        -------
        pd.DataFrame
            Two rows (one per target) with columns:
            ``target``, ``xgb_rmse``, ``xgb_mae``, ``xgb_r2``,
            ``lgb_rmse``, ``lgb_mae``, ``lgb_r2``.

        Requirements: 4.5, 8.5
        """
        if not self._use_lightgbm or self._lgb_model is None:
            raise RuntimeError(
                "compare_backends() requires use_lightgbm=True at construction time "
                "and a successful LightGBM installation."
            )

        X_test = self.test_df[self.feature_cols].values
        y_test = self.test_df[self.targets].values

        xgb_pred = self.model.predict(X_test)
        lgb_pred = self._lgb_model.predict(X_test)

        rows = []
        for i, target in enumerate(self.targets):
            y_t = y_test[:, i]
            rows.append(
                {
                    "target": target,
                    "xgb_rmse": float(np.sqrt(mean_squared_error(y_t, xgb_pred[:, i]))),
                    "xgb_mae": float(mean_absolute_error(y_t, xgb_pred[:, i])),
                    "xgb_r2": float(r2_score(y_t, xgb_pred[:, i])),
                    "lgb_rmse": float(np.sqrt(mean_squared_error(y_t, lgb_pred[:, i]))),
                    "lgb_mae": float(mean_absolute_error(y_t, lgb_pred[:, i])),
                    "lgb_r2": float(r2_score(y_t, lgb_pred[:, i])),
                }
            )

        return pd.DataFrame(rows, columns=[
            "target", "xgb_rmse", "xgb_mae", "xgb_r2",
            "lgb_rmse", "lgb_mae", "lgb_r2",
        ])

    def __len__(self) -> int:
        """Return the number of rows in the test set."""
        return len(self.test_df)
