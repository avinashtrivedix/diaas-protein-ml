"""
ingest_usda.py
--------------
Downloads, extracts, and parses the official USDA FoodData Central dataset.
Dynamically resolves nutrient IDs using nutrient.csv and pivots records
into a normalized Essential Amino Acid feature matrix (mg/g protein).
"""

import io
import logging
from pathlib import Path
import urllib.request
import zipfile
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

USDA_FOUNDATION_URL = "https://fdc.nal.usda.gov/fdc-datasets/FoodData_Central_foundation_food_csv_2024-04-18.zip"

CANONICAL_AA_NAMES = {
    "tryptophan": "Trp",
    "threonine": "Thr",
    "isoleucine": "Ile",
    "leucine": "Leu",
    "lysine": "Lys",
    "methionine": "Met",
    "cystine": "Cys",
    "cysteine": "Cys",
    "phenylalanine": "Phe",
    "tyrosine": "Tyr",
    "valine": "Val",
    "histidine": "His",
}


def download_and_extract_usda():
    """Downloads the USDA Foundation zip archive if not already cached."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    usda_dir = RAW_DIR / "usda_foundation"
    
    # If already downloaded and extracted, skip
    if (usda_dir / "food.csv").exists() and (usda_dir / "food_nutrient.csv").exists() and (usda_dir / "nutrient.csv").exists():
        logger.info("USDA Foundation dataset already exists locally. Skipping download.")
        return usda_dir

    logger.info("Downloading USDA Foundation Foods archive (~35MB)...")
    req = urllib.request.Request(USDA_FOUNDATION_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as response:
        zip_bytes = response.read()

    logger.info("Extracting CSVs to data/raw/usda_foundation/...")
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as z:
        z.extractall(usda_dir)
        
    return usda_dir


def build_dynamic_nutrient_map(usda_dir: Path) -> dict:
    """Reads nutrient.csv directly and dynamically maps IDs to standard names."""
    nutrient_file = list(usda_dir.glob("**/nutrient.csv"))[0]
    df_nutrients_meta = pd.read_csv(nutrient_file)

    id_to_name = {}
    for _, row in df_nutrients_meta.iterrows():
        n_id = int(row["id"])
        n_name = str(row["name"]).strip().lower()

        # Match total protein
        if n_name == "protein":
            id_to_name[n_id] = "Protein_g"
            continue

        # Match essential amino acids
        for chemical_name, standard_code in CANONICAL_AA_NAMES.items():
            if chemical_name in n_name:
                id_to_name[n_id] = standard_code
                break

    logger.info(f"Dynamically resolved {len(id_to_name)} target nutrient IDs from nutrient.csv")
    return id_to_name


def parse_and_pivot_amino_acids(usda_dir: Path) -> pd.DataFrame:
    """Loads raw USDA tables, filters target nutrients dynamically, and pivots to wide format."""
    food_file = list(usda_dir.glob("**/food.csv"))[0]
    nutrient_file = list(usda_dir.glob("**/food_nutrient.csv"))[0]

    logger.info("Reading food.csv and food_nutrient.csv...")
    df_food = pd.read_csv(food_file, usecols=["fdc_id", "description", "food_category_id"])
    df_nutrients = pd.read_csv(nutrient_file, usecols=["fdc_id", "nutrient_id", "amount"])

    # 1. Dynamically resolve nutrient map
    id_map = build_dynamic_nutrient_map(usda_dir)

    # 2. Filter for only matched nutrients
    df_filtered = df_nutrients[df_nutrients["nutrient_id"].isin(id_map.keys())].copy()
    df_filtered["nutrient_name"] = df_filtered["nutrient_id"].map(id_map)

    logger.info("Pivoting long-format nutrient records into wide feature matrix...")
    df_pivot = df_filtered.pivot_table(
        index="fdc_id",
        columns="nutrient_name",
        values="amount",
        aggfunc="mean"
    ).reset_index()

    # 3. Merge with food names
    df_merged = pd.merge(df_food, df_pivot, on="fdc_id", how="inner")

    # 4. Defensive checks: Ensure Protein_g exists and filter for protein-containing foods
    if "Protein_g" not in df_merged.columns:
        raise KeyError("Protein_g column was not generated during pivot.")

    df_valid = df_merged[df_merged["Protein_g"] >= 1.0].copy()

    eaa_cols = ["His", "Ile", "Leu", "Lys", "Met", "Cys", "Phe", "Tyr", "Thr", "Trp", "Val"]
    
    # Retain foods that have all essential amino acids recorded
    df_valid = df_valid.dropna(subset=eaa_cols)

    logger.info("Normalizing units to mg amino acid per gram of protein...")
    # USDA amounts are in g per 100g. Standardizing to mg AA / g Protein:
    for col in eaa_cols:
        df_valid[col] = (df_valid[col] / df_valid["Protein_g"]) * 1000.0

    # Derive composite families
    df_valid["SAA"] = df_valid["Met"] + df_valid["Cys"]
    df_valid["AAA"] = df_valid["Phe"] + df_valid["Tyr"]

    return df_valid


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    usda_dir = download_and_extract_usda()
    df_usda = parse_and_pivot_amino_acids(usda_dir)

    output_path = PROCESSED_DIR / "usda_amino_acid_profiles.csv"
    df_usda.to_csv(output_path, index=False)
    logger.info(f"Successfully processed {len(df_usda)} real USDA foods -> {output_path}")

    print("\n--- PROCESSED USDA AMINO ACID PROFILES (Sample) ---")
    cols_to_show = ["fdc_id", "description", "Protein_g", "Leu", "Lys", "SAA", "AAA"]
    print(df_usda[cols_to_show].head(5).to_string(index=False))


if __name__ == "__main__":
    main()