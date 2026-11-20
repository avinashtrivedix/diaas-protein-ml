import os
import sqlite3
import pandas as pd

# Ensure the data/ directory exists regardless of repo state
os.makedirs("data", exist_ok=True)

# 15-Ingredient Plant Protein Library (FAO/USDA baseline approximations)
data = {
    "Ingredient": [
        "Pea Isolate", "Rice Isolate", "Soy Isolate", "Hemp Seed", "Oat Protein",
        "Pumpkin Seed", "Lentil Isolate", "Chickpea Isolate", "Potato Protein",
        "Wheat Gluten", "Fava Bean Isolate", "Mung Bean Isolate",
        "Sunflower Seed", "Chia Seed", "Spirulina"
    ],
    "Category": [
        "Legume", "Grain", "Legume", "Seed", "Grain",
        "Seed", "Legume", "Legume", "Tuber",
        "Grain", "Legume", "Legume",
        "Seed", "Seed", "Algae"
    ],
    "Allergen": [
        "None", "None", "Soy", "None", "None",
        "None", "None", "None", "None",
        "Gluten", "None", "None",
        "None", "None", "None"
    ],
    "Digestibility": [0.89, 0.85, 0.90, 0.84, 0.90, 0.85, 0.82, 0.83, 0.92, 0.91, 0.87, 0.86, 0.85, 0.80, 0.84],
    "Cost_Per_Kg":   [12.0, 14.5, 10.0, 22.0, 18.0, 25.0, 15.0, 16.0, 24.0,  8.0, 14.0, 17.0, 16.5, 26.0, 32.0],
    "Calories_100g": [380,  370,  360,  400,  390,  410,  350,  360,  370,  370,  365,  360,  380,  410,  350],
    "Protein_100g":  [80,   80,   85,   50,   70,   60,   75,   70,   80,   75,   80,   80,   55,   45,   65],
    "His": [25.0, 23.0, 26.0, 28.0, 24.0, 27.0, 28.0, 28.0, 22.0, 21.0, 26.0, 27.0, 24.0, 26.0, 17.0],
    "Ile": [43.0, 41.0, 49.0, 38.0, 40.0, 42.0, 44.0, 43.0, 56.0, 38.0, 42.0, 45.0, 41.0, 40.0, 56.0],
    "Leu": [66.0, 82.0, 82.0, 66.0, 75.0, 73.0, 72.0, 70.0, 95.0, 70.0, 75.0, 78.0, 65.0, 63.0, 82.0],
    "Lys": [72.0, 31.0, 63.0, 38.0, 40.0, 35.0, 70.0, 68.0, 78.0, 17.0, 64.0, 69.0, 36.0, 45.0, 48.0],
    "SAA": [19.0, 38.0, 26.0, 41.0, 45.0, 30.0, 22.0, 26.0, 32.0, 40.0, 18.0, 20.0, 42.0, 38.0, 32.0],
    "AAA": [86.0, 85.0, 90.0, 80.0, 85.0, 82.0, 84.0, 83.0, 98.0, 88.0, 76.0, 86.0, 78.0, 80.0, 85.0],
    "Thr": [38.0, 37.0, 38.0, 34.0, 35.0, 32.0, 38.0, 38.0, 55.0, 26.0, 36.0, 34.0, 37.0, 35.0, 47.0],
    "Trp": [10.0, 11.0, 13.0, 12.0, 14.0, 15.0,  9.0, 10.0, 14.0,  9.0,  9.0, 10.0, 13.0, 16.0, 15.0],
    "Val": [50.0, 58.0, 50.0, 52.0, 55.0, 54.0, 52.0, 50.0, 65.0, 41.0, 48.0, 53.0, 51.0, 49.0, 62.0],
}

df = pd.DataFrame(data)

db_path = "data/formulation_engine.db"
with sqlite3.connect(db_path) as conn:
    df.to_sql("plant_proteins", conn, if_exists="replace", index=False)

print(f"Database '{db_path}' built with {len(df)} plant sources.")