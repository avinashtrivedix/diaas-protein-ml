"""
compositional_transforms.py
---------------------------
Applies Centered Log-Ratio (CLR) transformation to compositional
amino acid profiles to project simplex data into unconstrained Euclidean space.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"
FEATURES_DATA_DIR = BASE_DIR / "data" / "features"

EAA_COLS = ["His", "Ile", "Leu", "Lys", "Met", "Cys", "Phe", "Tyr", "Thr", "Trp", "Val"]


def centered_log_ratio_transform(df_eaa: pd.DataFrame, epsilon: float = 1e-5) -> pd.DataFrame:
    """
    Transforms amino acid mass profiles into CLR space.
    Formula: CLR(x_i) = ln(x_i / geometric_mean(x))
    """
    # 1. Add small epsilon to avoid log(0)
    data = df_eaa.values + epsilon

    # 2. Compute the geometric mean across rows (log-sum-exp trick)
    log_data = np.log(data)
    geometric_mean_log = np.mean(log_data, axis=1, keepdims=True)

    # 3. CLR = log(x_i) - log(geometric_mean)
    clr_matrix = log_data - geometric_mean_log

    clr_columns = [f"clr_{col}" for col in df_eaa.columns]
    return pd.DataFrame(clr_matrix, index=df_eaa.index, columns=clr_columns)


def main():
    input_path = PROCESSED_DATA_DIR / "food_diaas_calculated.csv"
    if not input_path.exists():
        raise FileNotFoundError("Run calculate_diaas.py first.")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded dataset with {len(df)} samples.")

    # Apply CLR transform on essential amino acids
    df_clr = centered_log_ratio_transform(df[EAA_COLS])

    # Merge original metadata and target with transformed features
    feature_df = pd.concat([
        df[["food_id", "name", "category", "processing", "true_ileal_digestibility", "calculated_diaas"]],
        df_clr
    ], axis=1)

    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FEATURES_DATA_DIR / "features_clr_transformed.csv"
    feature_df.to_csv(output_path, index=False)
    logger.info(f"Saved transformed feature matrix -> {output_path} (Shape: {feature_df.shape})")

    # Preview results
    print("\n--- CLR TRANSFORM PREVIEW (First 3 Rows) ---")
    print(feature_df[["name", "clr_Leu", "clr_Lys", "clr_Met", "calculated_diaas"]].head(3).to_string(index=False))


if __name__ == "__main__":
    main()