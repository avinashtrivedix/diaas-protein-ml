import os
import sqlite3
import numpy as np
import pandas as pd
from scipy.optimize import linprog

FAO_REFERENCE = {
    "His": 16.0, "Ile": 30.0, "Leu": 59.0, "Lys": 45.0,
    "SAA": 22.0, "AAA": 38.0, "Thr": 23.0, "Trp": 6.0, "Val": 39.0
}

def load_plant_library(db_path = "data/formulation_engine.db", excluded_allergens = None):
    """Loads plant protein ingridinets from SQLITE with optional Allergen filtering."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found at '{db_path}'. Run 'python setup_db.py' first."
        )
    query = "Select * FROM plant_proteins"
    params = []

    if excluded_allergens:
        placeholders = ", ".join(["?"] * len(excluded_allergens))
        query += f" WHERE Allergen NOT IN ({placeholders})"
        params.extend(excluded_allergens)

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(query, conn, params = params)

    return df.set_index("Ingredient").to_dict(orient="index")

def optimize_protein_blend(
    ingredients,
    fao_reference=FAO_REFERENCE,
    objective="max_diaas",
    min_diaas=1.0,
    bounds_config= None,
    min_leucine=None,
    target_protein_g=25.0,
):
    """
    Finds the optimal plant protein blend using Linear Programming.
        - "max_diaas": Maximize the limiting digestible amino acid ratio (t).
        - "min_cost": Minimize cost per serving while keeping DIAAS >= min_diaas.
        - "min_calories": Minimize calories per serving while keeping DIAAS >= min_diaas.
    """
    if not ingredients:
        return {"status": "Infeasible - No ingredients available after filtering", "score": 0.0, "formulation": {}}
    
    ingredient_names = list(ingredients.keys())
    aa_keys = list(fao_reference.keys())
    n_ing = len(ingredient_names)
    n_aa = len(aa_keys)

    # Digestible amino acid matrix: shape (n_ing, n_aa)
    # Accessing "Digestibility" directly so schema mismatches fail loudly
    A = np.array([
        [ingredients[ing][aa] * ingredients[ing]["Digestibility"] for aa in aa_keys]
        for ing in ingredient_names
    ])
    fao_vec = np.array([fao_reference[aa] for aa in aa_keys])

    bounds_config = bounds_config or {}
    ing_bounds = [bounds_config.get(name, (0.0, 1.0)) for name in ingredient_names]

    if objective == "max_diaas":
        # Decision variables: [w_1, ..., w_n, t]
        c = np.zeros(n_ing + 1)
        c[-1] = -1.0  # Minimize -t  <=>  Maximize t

        # -sum(w_i * digestible_AA_ij / FAO_j) + t <= 0
        A_ub = np.zeros((n_aa, n_ing + 1))
        for j in range(n_aa):
            A_ub[j, :n_ing] = -A[:, j] / fao_vec[j]
            A_ub[j, -1] = 1.0
        b_ub = np.zeros(n_aa)

        A_eq = np.zeros((1, n_ing + 1))
        A_eq[0, :n_ing] = 1.0
        b_eq = np.array([1.0])

        bounds = ing_bounds + [(0.0, None)]

    elif objective in ("min_cost", "min_calories"):
        # Decision variables: [w_1, ..., w_n] (no 't' needed; min_diaas is a fixed floor)
        protein_fracs = np.array([ingredients[ing]["Protein_100g"] / 100.0 for ing in ingredient_names])

        if objective == "min_cost":
            # Cost per gram of protein
            c = np.array([ingredients[ing]["Cost_Per_Kg"] / 1000.0 for ing in ingredient_names]) / protein_fracs
        else:
            # Calories per gram of protein
            c = np.array([ingredients[ing]["Calories_100g"] / 100.0 for ing in ingredient_names]) / protein_fracs

        # -sum(w_i * digestible_AA_ij / FAO_j) <= -min_diaas
        A_ub = np.zeros((n_aa, n_ing))
        for j in range(n_aa):
            A_ub[j, :n_ing] = -A[:, j] / fao_vec[j]
        b_ub = np.full(n_aa, -float(min_diaas))

        A_eq = np.ones((1, n_ing))
        b_eq = np.array([1.0])
        bounds = ing_bounds

    else:
        raise ValueError(f"Unknown objective '{objective}'. Choose 'max_diaas', 'min_cost', or 'min_calories'.")
    
    # Optional dynamic constraint: minimum digestible Leucine (mg/g protein)
    if min_leucine is not None:
        leu_idx = aa_keys.index("Leu")
        leu_row = np.zeros(A_ub.shape[1])
        leu_row[:n_ing] = -A[:, leu_idx]
        A_ub = np.vstack([A_ub, leu_row])
        b_ub = np.append(b_ub, -float(min_leucine))

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if not res.success:
        return {"status": "Infeasible - Constraints are too strict", "score": 0.0, "formulation": {}}

    weights = res.x[:n_ing]
    blend_aa = weights @ A
    achieved_diaas = float(np.min(blend_aa / fao_vec))
    blend_profile = {aa: round(float(val), 2) for aa, val in zip(aa_keys, blend_aa)}

    # Convert protein shares into physical scoop metrics for target_protein_g
    powder_grams = {}
    total_cost = 0.0
    total_calories = 0.0

    for name, w in zip(ingredient_names, weights):
        if w < 1e-4:
            continue
        p_frac = ingredients[name]["Protein_100g"] / 100.0
        grams = (w * target_protein_g) / p_frac
        powder_grams[name] = round(float(grams), 2)
        total_cost += (grams / 1000.0) * ingredients[name]["Cost_Per_Kg"]
        total_calories += (grams / 100.0) * ingredients[name]["Calories_100g"]

    total_scoop_g = sum(powder_grams.values())
    powder_pct = {
        name: round((g / total_scoop_g) * 100.0, 2)
        for name, g in powder_grams.items()
    }
    return {
        "status": "Optimal",
        "objective": objective,
        "score": round(achieved_diaas, 3),
        "formulation": powder_pct,          # % by physical powder weight (what you actually scoop)
        "powder_grams": powder_grams,       # Exact grams per serving
        "blend_profile": blend_profile,     # Digestible mg/g protein for the Radar Chart
        "metrics": {
            "target_protein_g": target_protein_g,
            "scoop_size_g": round(total_scoop_g, 1),
            "cost_per_serving": round(total_cost, 2),
            "calories_per_serving": round(total_calories, 1),
        },
    }

if __name__ == "__main__":
    # Verify all 3 modes directly against SQLite
    library = load_plant_library(excluded_allergens=["Soy"])

    for mode in ["max_diaas", "min_cost", "min_calories"]:
        res = optimize_protein_blend(library, objective=mode, min_diaas=1.0)
        print(f"\n=== Mode: {mode} (Soy-Free) ===")
        print(f"Status: {res['status']} | DIAAS: {res['score']}")
        if res["status"] == "Optimal":
            m = res["metrics"]
            print(f"25g Protein Serving -> Scoop: {m['scoop_size_g']}g | Cost: ${m['cost_per_serving']} | Calories: {m['calories_per_serving']} kcal")
            for ing, pct in res["formulation"].items():
                print(f"  - {ing}: {pct}% ({res['powder_grams'][ing]}g)")

