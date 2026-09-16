import sqlite3
import pandas as pd

# The Raw Blueprint (Expanded Plant Library)
# Cost is representd in a generic numeric weight (eg . price per kg)
data = {
    "Ingredient": ["Pea Isolate", "Rice Isolate", "Soy Isolate", "Hemp Seed", "Oat Protein", "Pumpkin Seed", "Lentil Isolate", "Chickpea Isolate"],
    "Digestibility": [0.89, 0.85, 0.90, 0.84, 0.90, 0.85, 0.82, 0.83],
    "Cost_Per_Kg": [12.0, 14.5, 10.0, 22.0, 18.0, 25.0, 15.0, 16.0], 
    "Calories_100g": [380, 370, 360, 400, 390, 410, 350, 360],
    "Protein_100g": [80, 80, 85, 50, 70, 60, 75, 70],
    "His": [25.0, 23.0, 26.0, 28.0, 24.0, 27.0, 28.0, 28.0],
    "Ile": [43.0, 41.0, 49.0, 38.0, 40.0, 42.0, 44.0, 43.0],
    "Leu": [66.0, 82.0, 82.0, 66.0, 75.0, 73.0, 72.0, 70.0],
    "Lys": [72.0, 31.0, 63.0, 38.0, 40.0, 35.0, 70.0, 68.0],
    "SAA": [19.0, 38.0, 26.0, 41.0, 45.0, 30.0, 22.0, 26.0],
    "AAA": [86.0, 85.0, 90.0, 80.0, 85.0, 82.0, 84.0, 83.0],
    "Thr": [38.0, 37.0, 38.0, 34.0, 35.0, 32.0, 38.0, 38.0],
    "Trp": [10.0, 11.0, 13.0, 12.0, 14.0, 15.0, 9.0, 10.0],
    "Val": [50.0, 58.0, 50.0, 52.0, 55.0, 54.0, 52.0, 50.0]
}

# 2. Transform into a Pandas DataFrame for safe handling
df = pd.DataFrame(data)

# 3. Architect the SQLite Database
db_path = "data/formulation_engine.db"
conn = sqlite3.connect(db_path)

# 4. Write the dataframe to a SQL table named 'plant_proteins'
df.to_sql("plant_proteins", conn, if_exists="replace", index=False)
conn.close()

print(f"✅ Enterprise Database '{db_path}' successfully generated with {len(df)} plant sources.")