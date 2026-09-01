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
    # Add epsilon to prevent ln(0) errors
    data = df_eaa.values + epsilon

    # Compute row-wise geometric mean using log-average
    log_data = np.log(data)
    geometric_mean_log = np.mean(log_data, axis=1, keepdims=True)

    # CLR = ln(x_i) - ln(geometric_mean)
    clr_matrix = log_data - geometric_mean_log

    clr_columns = [f"clr_{col}" for col in df_eaa.columns]
    return pd.DataFrame(clr_matrix, index=df_eaa.index, columns=clr_columns)


def main():
    input_path = PROCESSED_DATA_DIR / "usda_amino_acid_profiles.csv"
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input: {input_path}")

    df = pd.read_csv(input_path)
    logger.info(f"Loaded USDA dataset with {len(df)} food records.")

    # 1. Transform only the essential amino acid columns
    df_clr = centered_log_ratio_transform(df[EAA_COLS])

    # 2. Dynamically pick metadata columns that exist in the file (defensive engineering)
    metadata_candidates = ["fdc_id", "description", "food_category_id", "Protein_g", "SAA", "AAA"]
    meta_cols = [c for c in metadata_candidates if c in df.columns]

    # 3. Combine metadata with the new CLR feature columns
    feature_df = pd.concat([df[meta_cols], df_clr], axis=1)

    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = FEATURES_DATA_DIR / "features_usda_clr.csv"
    feature_df.to_csv(output_path, index=False)
    logger.info(f"Saved CLR features -> {output_path} (Shape: {feature_df.shape})")

    # Preview top rows
    print("\n--- USDA CLR FEATURE MATRIX (First 3 Rows) ---")
    preview_cols = ["description", "clr_Leu", "clr_Lys", "clr_Val"]
    print(feature_df[preview_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()