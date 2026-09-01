"""
train_baseline.py
-----------------
Trains baseline Ridge Regression and Random Forest models on CLR features
to evaluate predictive power on protein quality metrics.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
FEATURE_PATH = BASE_DIR / "data" / "features" / "features_usda_clr.csv"


def evaluate_model_cv(model, X: np.ndarray, y: np.ndarray, n_splits: int = 5):
    """Evaluates a regression model using K-Fold Cross-Validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_true_all, y_pred_all = [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        y_true_all.extend(y_test)
        y_pred_all.extend(pred)

    y_true_all, y_pred_all = np.array(y_true_all), np.array(y_pred_all)
    mae = mean_absolute_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    r2 = r2_score(y_true_all, y_pred_all)

    return mae, rmse, r2


def main():
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing feature file: {FEATURE_PATH}")

    df = pd.read_csv(FEATURE_PATH)
    logger.info(f"Loaded USDA CLR feature matrix: {df.shape}")

    # Extract all CLR feature columns
    feature_cols = [c for c in df.columns if c.startswith("clr_")]
    X = df[feature_cols].values
    
    # We evaluate prediction of total crude protein concentration as our initial proxy target
    y = df["Protein_g"].values

    models = {
        "Ridge Regression (L2 Regularized)": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=4, random_state=42)
    }

    print("\n================ BASELINE MODEL BENCHMARK (5-Fold CV) ================")
    for name, model in models.items():
        mae, rmse, r2 = evaluate_model_cv(model, X, y, n_splits=5)
        print(f"\nModel: {name}")
        print(f"  Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  R² Score (Variance Explained): {r2:.4f}")
    print("=======================================================================\n")


if __name__ == "__main__":
    main()