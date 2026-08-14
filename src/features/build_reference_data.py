"""
build_reference_data.py
-----------------------
Ingests, validates, and serializes the official FAO/WHO 2013 Amino Acid
Reference Scoring Patterns for DIAAS calculation.
"""

import json
import logging
from pathlib import Path
import pandas as pd

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Define Core Storage Paths
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# FAO/WHO 2013 Expert Consultation Reference Patterns (mg / g protein)
FAO_WHO_2013_PATTERNS = {
    "infant_0_6m": {
        "description": "Infant (0 to 6 months) based on human milk composition",
        "requirements_mg_per_g": {
            "His": 21.0,
            "Ile": 55.0,
            "Leu": 96.0,
            "Lys": 69.0,
            "Met": 17.0,
            "Cys": 20.0,
            "SAA": 37.0,  # Sulfur Amino Acids (Met + Cys)
            "Phe": 40.0,
            "Tyr": 48.0,
            "AAA": 88.0,  # Aromatic Amino Acids (Phe + Tyr)
            "Thr": 44.0,
            "Trp": 17.0,
            "Val": 55.0,
        }
    },
    "child_6m_3y": {
        "description": "Young Children (6 months to 3 years)",
        "requirements_mg_per_g": {
            "His": 18.0,
            "Ile": 31.0,
            "Leu": 63.0,
            "Lys": 52.0,
            "Met": 14.0,
            "Cys": 13.0,
            "SAA": 27.0,
            "Phe": 26.0,
            "Tyr": 20.0,
            "AAA": 46.0,
            "Thr": 31.0,
            "Trp": 8.5,
            "Val": 41.0,
        }
    },
    "older_child_adult": {
        "description": "Older Children, Adolescents, and Adults (> 3 years) [Standard Benchmark]",
        "requirements_mg_per_g": {
            "His": 16.0,
            "Ile": 30.0,
            "Leu": 61.0,
            "Lys": 48.0,
            "Met": 16.0,
            "Cys": 7.0,
            "SAA": 23.0,
            "Phe": 27.0,
            "Tyr": 14.0,
            "AAA": 41.0,
            "Thr": 25.0,
            "Trp": 6.6,
            "Val": 40.0,
        }
    }
}


def validate_scoring_patterns(data: dict) -> None:
    """Defensive schema & domain constraint validation."""
    logger.info("Executing defensive schema validation on reference patterns...")
    
    for age_group, payload in data.items():
        reqs = payload["requirements_mg_per_g"]
        
        # 1. Non-negativity constraint
        for aa, val in reqs.items():
            if val <= 0:
                raise ValueError(f"Requirement for {aa} in {age_group} must be > 0. Got: {val}")
        
        # 2. Combined Amino Acid Additivity Check (SAA = Met + Cys; AAA = Phe + Tyr)
        calculated_saa = reqs["Met"] + reqs["Cys"]
        calculated_aaa = reqs["Phe"] + reqs["Tyr"]
        
        if abs(calculated_saa - reqs["SAA"]) > 1e-3:
            raise ValueError(
                f"SAA mismatch in {age_group}: Met ({reqs['Met']}) + Cys ({reqs['Cys']}) "
                f"!= SAA ({reqs['SAA']})"
            )
            
        if abs(calculated_aaa - reqs["AAA"]) > 1e-3:
            raise ValueError(
                f"AAA mismatch in {age_group}: Phe ({reqs['Phe']}) + Tyr ({reqs['Tyr']}) "
                f"!= AAA ({reqs['AAA']})"
            )
            
    logger.info("Schema validation passed successfully: All physical invariants verified.")


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Run schema validation
    validate_scoring_patterns(FAO_WHO_2013_PATTERNS)

    # 2. Save immutable Raw JSON snapshot
    raw_json_path = RAW_DATA_DIR / "fao_who_2013_reference_patterns.json"
    with open(raw_json_path, "w", encoding="utf-8") as f:
        json.dump(FAO_WHO_2013_PATTERNS, f, indent=4)
    logger.info(f"Raw reference snapshot saved -> {raw_json_path}")

    # 3. Flatten and export clean tabular reference dataframe
    rows = []
    for age_group, payload in FAO_WHO_2013_PATTERNS.items():
        row = {"age_group": age_group, "description": payload["description"]}
        row.update(payload["requirements_mg_per_g"])
        rows.append(row)

    df_reference = pd.DataFrame(rows)
    processed_csv_path = PROCESSED_DATA_DIR / "reference_scoring_patterns.csv"
    df_reference.to_csv(processed_csv_path, index=False)
    logger.info(f"Processed tabular scoring patterns saved -> {processed_csv_path}")


if __name__ == "__main__":
    main()