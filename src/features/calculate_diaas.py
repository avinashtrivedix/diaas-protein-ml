"""
calculate_diaas.py
------------------
Computes Amino Acid Ratios, identifies the rate-limiting amino acid,
and calculates true DIAAS scores against WHO reference patterns.
"""

from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# The essential amino acids evaluated in the scoring pattern
EAA_COLUMNS = ["His", "Ile", "Leu", "Lys", "SAA", "AAA", "Thr", "Trp", "Val"]


def calculate_diaas_for_dataset(df_foods: pd.DataFrame, df_ref: pd.DataFrame, target_age_group: str = "older_child_adult") -> pd.DataFrame:
    """
    Vectorized calculation of amino acid ratios, limiting amino acid, and DIAAS score.
    """
    # 1. Extract the reference pattern for the target age group
    ref_row = df_ref[df_ref["age_group"] == target_age_group].iloc[0]
    
    # 2. Compute the ratio for each essential amino acid
    ratio_df = pd.DataFrame(index=df_foods.index)
    for aa in EAA_COLUMNS:
        req_val = ref_row[aa]
        ratio_df[f"ratio_{aa}"] = df_foods[aa] / req_val

    # 3. Find the lowest ratio and the limiting amino acid for each food
    df_result = df_foods.copy()
    
    # Concatenate ratio columns to results
    df_result = pd.concat([df_result, ratio_df], axis=1)
    
    # Find minimum ratio per row
    df_result["lowest_eaa_ratio"] = ratio_df.min(axis=1)
    
    # Identify which amino acid produced that minimum (strip the 'ratio_' prefix)
    df_result["limiting_amino_acid"] = ratio_df.idxmin(axis=1).str.replace("ratio_", "")
    
    # 4. DIAAS Score = Lowest Ratio * True Ileal Digestibility
    df_result["calculated_diaas"] = df_result["lowest_eaa_ratio"] * df_result["true_ileal_digestibility"]

    return df_result


def main():
    food_path = PROCESSED_DATA_DIR / "food_amino_acid_profiles.csv"
    ref_path = PROCESSED_DATA_DIR / "reference_scoring_patterns.csv"

    if not food_path.exists() or not ref_path.exists():
        raise FileNotFoundError("Missing processed input files. Run build_reference_data.py and build_food_dataset.py first.")

    logger.info("Loading processed food profiles and reference patterns...")
    df_foods = pd.read_csv(food_path)
    df_ref = pd.read_csv(ref_path)

    logger.info("Computing DIAAS scores and rate-limiting amino acids...")
    df_diaas = calculate_diaas_for_dataset(df_foods, df_ref)

    # Save calculated dataset
    output_path = PROCESSED_DATA_DIR / "food_diaas_calculated.csv"
    df_diaas.to_csv(output_path, index=False)
    logger.info(f"Calculated DIAAS dataset saved -> {output_path} (Shape: {df_diaas.shape})")

    # Display a quick preview table of key outputs in terminal
    preview_cols = ["name", "limiting_amino_acid", "lowest_eaa_ratio", "true_ileal_digestibility", "calculated_diaas", "published_diaas_adult"]
    print("\n--- CALCULATION RESULTS PREVIEW ---")
    print(df_diaas[preview_cols].to_string(index=False))


if __name__ == "__main__":
    main()