"""
build_food_dataset.py
---------------------
Compiles, validates, and serializes a curated benchmark dataset of food proteins,
including their 9 Essential Amino Acid (EAA) distributions (mg/g protein),
anti-nutritional factor metrics, processing types, and ground-truth ileal digestibilities.
"""

import json
import logging
from pathlib import Path
import pandas as pd
import numpy as np

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Benchmark Curated Dataset: Amino Acids in mg / g of protein
# Digestibility values based on FAO / literature standard ileal digestibility trials
FOOD_PROTEIN_DATABASE = [
    # --- Animal / Dairy Proteins ---
    {
        "food_id": "FP_001",
        "name": "Whey Protein Isolate",
        "category": "Animal_Dairy",
        "processing": "Isolate",
        "crude_protein_pct": 90.0,
        "His": 19.0, "Ile": 64.0, "Leu": 118.0, "Lys": 98.0,
        "Met": 22.0, "Cys": 27.0, "Phe": 32.0, "Tyr": 33.0,
        "Thr": 69.0, "Trp": 21.0, "Val": 58.0,
        "phytate_mg_g": 0.0, "trypsin_inhibitor_tiu_g": 0.0, "tannins_mg_g": 0.0,
        "true_ileal_digestibility": 0.98,
        "published_diaas_adult": 1.15
    },
    {
        "food_id": "FP_002",
        "name": "Micellar Casein",
        "category": "Animal_Dairy",
        "processing": "Concentrate",
        "crude_protein_pct": 82.0,
        "His": 28.0, "Ile": 52.0, "Leu": 95.0, "Lys": 80.0,
        "Met": 28.0, "Cys": 6.0, "Phe": 51.0, "Tyr": 54.0,
        "Thr": 45.0, "Trp": 14.0, "Val": 64.0,
        "phytate_mg_g": 0.0, "trypsin_inhibitor_tiu_g": 0.0, "tannins_mg_g": 0.0,
        "true_ileal_digestibility": 0.96,
        "published_diaas_adult": 1.18
    },
    {
        "food_id": "FP_003",
        "name": "Whole Egg Powder",
        "category": "Animal_Poultry",
        "processing": "Cooked",
        "crude_protein_pct": 48.0,
        "His": 24.0, "Ile": 54.0, "Leu": 86.0, "Lys": 72.0,
        "Met": 32.0, "Cys": 23.0, "Phe": 54.0, "Tyr": 40.0,
        "Thr": 47.0, "Trp": 16.0, "Val": 66.0,
        "phytate_mg_g": 0.0, "trypsin_inhibitor_tiu_g": 0.0, "tannins_mg_g": 0.0,
        "true_ileal_digestibility": 0.97,
        "published_diaas_adult": 1.13
    },
    {
        "food_id": "FP_004",
        "name": "Cooked Beef (Lean)",
        "category": "Animal_Meat",
        "processing": "Cooked",
        "crude_protein_pct": 28.0,
        "His": 34.0, "Ile": 46.0, "Leu": 81.0, "Lys": 85.0,
        "Met": 26.0, "Cys": 12.0, "Phe": 40.0, "Tyr": 33.0,
        "Thr": 44.0, "Trp": 11.0, "Val": 51.0,
        "phytate_mg_g": 0.0, "trypsin_inhibitor_tiu_g": 0.0, "tannins_mg_g": 0.0,
        "true_ileal_digestibility": 0.94,
        "published_diaas_adult": 1.10
    },
    {
        "food_id": "FP_005",
        "name": "Cooked Chicken Breast",
        "category": "Animal_Poultry",
        "processing": "Cooked",
        "crude_protein_pct": 31.0,
        "His": 31.0, "Ile": 47.0, "Leu": 79.0, "Lys": 88.0,
        "Met": 27.0, "Cys": 13.0, "Phe": 39.0, "Tyr": 34.0,
        "Thr": 43.0, "Trp": 12.0, "Val": 50.0,
        "phytate_mg_g": 0.0, "trypsin_inhibitor_tiu_g": 0.0, "tannins_mg_g": 0.0,
        "true_ileal_digestibility": 0.95,
        "published_diaas_adult": 1.08
    },

    # --- Plant Isolates & Concentrates ---
    {
        "food_id": "FP_006",
        "name": "Soy Protein Isolate",
        "category": "Plant_Legume",
        "processing": "Isolate",
        "crude_protein_pct": 88.0,
        "His": 26.0, "Ile": 49.0, "Leu": 78.0, "Lys": 63.0,
        "Met": 13.0, "Cys": 13.0, "Phe": 52.0, "Tyr": 38.0,
        "Thr": 38.0, "Trp": 13.0, "Val": 50.0,
        "phytate_mg_g": 12.5, "trypsin_inhibitor_tiu_g": 2.1, "tannins_mg_g": 0.8,
        "true_ileal_digestibility": 0.91,
        "published_diaas_adult": 0.90
    },
    {
        "food_id": "FP_007",
        "name": "Pea Protein Isolate",
        "category": "Plant_Legume",
        "processing": "Isolate",
        "crude_protein_pct": 82.0,
        "His": 25.0, "Ile": 45.0, "Leu": 84.0, "Lys": 72.0,
        "Met": 11.0, "Cys": 10.0, "Phe": 55.0, "Tyr": 37.0,
        "Thr": 38.0, "Trp": 10.0, "Val": 50.0,
        "phytate_mg_g": 14.0, "trypsin_inhibitor_tiu_g": 1.8, "tannins_mg_g": 1.2,
        "true_ileal_digestibility": 0.89,
        "published_diaas_adult": 0.82
    },
    {
        "food_id": "FP_008",
        "name": "Brown Rice Protein Concentrate",
        "category": "Plant_Cereal",
        "processing": "Concentrate",
        "crude_protein_pct": 78.0,
        "His": 23.0, "Ile": 43.0, "Leu": 82.0, "Lys": 31.0,  # Lysine deficient
        "Met": 29.0, "Cys": 24.0, "Phe": 56.0, "Tyr": 50.0,
        "Thr": 37.0, "Trp": 14.0, "Val": 60.0,
        "phytate_mg_g": 18.2, "trypsin_inhibitor_tiu_g": 0.5, "tannins_mg_g": 2.0,
        "true_ileal_digestibility": 0.87,
        "published_diaas_adult": 0.58
    },

    # --- Whole Plant Sources (Raw / Cooked) ---
    {
        "food_id": "FP_009",
        "name": "Cooked Chickpeas",
        "category": "Plant_Legume",
        "processing": "Cooked",
        "crude_protein_pct": 8.9,
        "His": 27.0, "Ile": 42.0, "Leu": 71.0, "Lys": 66.0,
        "Met": 13.0, "Cys": 14.0, "Phe": 53.0, "Tyr": 31.0,
        "Thr": 37.0, "Trp": 9.0, "Val": 44.0,
        "phytate_mg_g": 8.4, "trypsin_inhibitor_tiu_g": 3.4, "tannins_mg_g": 2.5,
        "true_ileal_digestibility": 0.78,
        "published_diaas_adult": 0.74
    },
    {
        "food_id": "FP_010",
        "name": "Cooked Lentils",
        "category": "Plant_Legume",
        "processing": "Cooked",
        "crude_protein_pct": 9.0,
        "His": 25.0, "Ile": 43.0, "Leu": 75.0, "Lys": 71.0,
        "Met": 9.0, "Cys": 11.0, "Phe": 51.0, "Tyr": 33.0,
        "Thr": 39.0, "Trp": 9.5, "Val": 49.0,
        "phytate_mg_g": 9.8, "trypsin_inhibitor_tiu_g": 4.1, "tannins_mg_g": 3.2,
        "true_ileal_digestibility": 0.76,
        "published_diaas_adult": 0.63
    },
    {
        "food_id": "FP_011",
        "name": "Whole Wheat Flour",
        "category": "Plant_Cereal",
        "processing": "Raw",
        "crude_protein_pct": 13.2,
        "His": 24.0, "Ile": 37.0, "Leu": 67.0, "Lys": 26.0,  # Severely Lysine deficient
        "Met": 17.0, "Cys": 23.0, "Phe": 48.0, "Tyr": 31.0,
        "Thr": 28.0, "Trp": 12.0, "Val": 45.0,
        "phytate_mg_g": 22.0, "trypsin_inhibitor_tiu_g": 1.2, "tannins_mg_g": 1.9,
        "true_ileal_digestibility": 0.83,
        "published_diaas_adult": 0.45
    },
    {
        "food_id": "FP_012",
        "name": "Cooked Quinoa",
        "category": "Plant_Pseudocereal",
        "processing": "Cooked",
        "crude_protein_pct": 4.4,
        "His": 30.0, "Ile": 41.0, "Leu": 68.0, "Lys": 58.0,  # Well balanced for plant
        "Met": 22.0, "Cys": 18.0, "Phe": 46.0, "Tyr": 36.0,
        "Thr": 38.0, "Trp": 13.0, "Val": 52.0,
        "phytate_mg_g": 6.1, "trypsin_inhibitor_tiu_g": 0.2, "tannins_mg_g": 1.1,
        "true_ileal_digestibility": 0.84,
        "published_diaas_adult": 0.81
    }
]


def validate_and_compute_composites(df: pd.DataFrame) -> pd.DataFrame:
    """Computes composite amino acid families (SAA, AAA) and verifies non-negativity."""
    logger.info("Verifying food dataset constraints and deriving composite features...")
    
    # Calculate SAA (Sulfur Amino Acids) and AAA (Aromatic Amino Acids)
    df["SAA"] = df["Met"] + df["Cys"]
    df["AAA"] = df["Phe"] + df["Tyr"]

    # Calculate Total Essential Amino Acids (Total EAA)
    eaa_cols = ["His", "Ile", "Leu", "Lys", "Met", "Cys", "Phe", "Tyr", "Thr", "Trp", "Val"]
    df["Total_EAA_mg_g"] = df[eaa_cols].sum(axis=1)

    # Validate physical non-negativity
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if (df[numeric_cols] < 0).any().any():
        raise ValueError("Detected negative concentrations in numeric fields.")

    logger.info("Food dataset validated successfully.")
    return df


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save Raw JSON
    raw_json_path = RAW_DATA_DIR / "raw_food_proteins.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(FOOD_PROTEIN_DATABASE, f, indent=4)
    logger.info(f"Raw food database snapshot saved -> {raw_json_path}")

    # 2. Process and save tabular dataset
    df_raw = pd.DataFrame(FOOD_PROTEIN_DATABASE)
    df_processed = validate_and_compute_composites(df_raw)

    processed_csv_path = PROCESSED_DATA_DIR / "food_amino_acid_profiles.csv"
    df_processed.to_csv(processed_csv_path, index=False)
    logger.info(f"Processed food dataset saved -> {processed_csv_path} (Shape: {df_processed.shape})")


if __name__ == "__main__":
    main()