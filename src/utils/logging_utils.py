# =============================================================================
# src/utils/logging_utils.py — OP'26 logging helpers
# =============================================================================

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> None:
    """Set up root logger with timestamp format."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
        force=True,
    )


def log_dependency_versions() -> None:
    """Log key dependency versions at INFO level for reproducibility."""
    logger = logging.getLogger("ev_agentic.versions")
    try:
        import pandas as pd
        logger.info("pandas     %s", pd.__version__)
    except ImportError:
        logger.warning("pandas not installed")

    try:
        import numpy as np
        logger.info("numpy      %s", np.__version__)
    except ImportError:
        logger.warning("numpy not installed")

    try:
        import xgboost as xgb
        logger.info("xgboost    %s", xgb.__version__)
    except ImportError:
        logger.warning("xgboost not installed")

    try:
        import sklearn
        logger.info("sklearn    %s", sklearn.__version__)
    except ImportError:
        logger.warning("scikit-learn not installed")

    try:
        import lightgbm as lgb
        logger.info("lightgbm   %s", lgb.__version__)
    except ImportError:
        pass  # optional dependency — silence if absent
