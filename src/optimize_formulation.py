import numpy as np
from scipy.optimize import linprog

def optimize_protein_blend(ingredients, fao_reference, bounds_config=None, min_leucine=None):
    """
    Dynamically finds the optimal plant protein blend given custom constraints.
    
    bounds_config: dict of tuples, e.g., {"Soy Protein Isolate": (0.0, 0.0), "Pea Protein Isolate": (0.10, 0.30)}
    min_leucine: float, minimum mg of Leucine per g of protein required in the final blend.
    """
    ingredient_names = list(ingredients.keys())
    aa_keys = list(fao_reference.keys())
    
    n_ingredients = len(ingredient_names)
    n_aa = len(aa_keys)

    # Build matrix A (shape: n_ingredients x n_aa)
    A = np.array([[ingredients[ing][aa] for aa in aa_keys] for ing in ingredient_names])
    fao_vec = np.array([fao_reference[aa] for aa in aa_keys])

    # Objective: Minimize -t (Maximize AAS)
    c = np.zeros(n_ingredients + 1)
    c[-1] = -1.0

    # Base Constraints: Target Score <= Actual Amino Acids
    A_ub = np.zeros((n_aa, n_ingredients + 1))
    for j in range(n_aa):
        for i in range(n_ingredients):
            A_ub[j, i] = -A[i, j] / fao_vec[j]
        A_ub[j, -1] = 1.0 
    b_ub = np.zeros(n_aa)

    # Dynamic Constraint: Leucine Threshold
    if min_leucine is not None:
        leu_index = aa_keys.index("Leu")
        leu_constraint = np.zeros(n_ingredients + 1)
        for i in range(n_ingredients):
            leu_constraint[i] = -A[i, leu_index]
        A_ub = np.vstack([A_ub, leu_constraint])
        b_ub = np.append(b_ub, -min_leucine)

    # Equality Constraint: Sum of weights = 100%
    A_eq = np.zeros((1, n_ingredients + 1))
    A_eq[0, :n_ingredients] = 1.0
    b_eq = np.array([1.0])

    # Dynamic Bounds Configuration
    bounds_config = bounds_config or {}
    bounds = []
    for name in ingredient_names:
        # Default to (0, 1) if no custom bound is provided for this ingredient
        bounds.append(bounds_config.get(name, (0.0, 1.0)))
    bounds.append((0, None)) # Bound for the target score (t)

    # Solve
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    
    if not res.success:
        return {"status": "Infeasible - Constraints are too strict", "score": 0, "formulation": {}}

    formulation = {name: round(weight * 100, 2) for name, weight in zip(ingredient_names, res.x[:n_ingredients])}
    
    return {
        "status": "Optimal",
        "score": round(res.x[-1], 3),
        "formulation": formulation
    }

# ==========================================
# TEST THE DYNAMIC ENGINE
# ==========================================
if __name__ == "__main__":
    fao_ref = {
        "His": 16.0, "Ile": 30.0, "Leu": 59.0, "Lys": 45.0,
        "SAA": 22.0, "AAA": 38.0, "Thr": 23.0, "Trp": 6.0, "Val": 39.0
    }
    
    plant_sources = {
        "Pea Protein Isolate": {"His": 25.0, "Ile": 43.0, "Leu": 66.0, "Lys": 72.0, "SAA": 19.0, "AAA": 86.0, "Thr": 38.0, "Trp": 10.0, "Val": 50.0},
        "Brown Rice Protein": {"His": 23.0, "Ile": 41.0, "Leu": 82.0, "Lys": 31.0, "SAA": 38.0, "AAA": 85.0, "Thr": 37.0, "Trp": 11.0, "Val": 58.0},
        "Soy Protein Isolate": {"His": 26.0, "Ile": 49.0, "Leu": 82.0, "Lys": 63.0, "SAA": 26.0, "AAA": 90.0, "Thr": 38.0, "Trp": 13.0, "Val": 50.0},
        "Hemp Seed Protein": {"His": 28.0, "Ile": 38.0, "Leu": 66.0, "Lys": 38.0, "SAA": 41.0, "AAA": 80.0, "Thr": 34.0, "Trp": 12.0, "Val": 52.0}
    }

    # Scenario: Client wants Soy locked to 0%, and Pea between 10% and 30%.
    custom_bounds = {
        "Soy Protein Isolate": (0.0, 0.0),
        "Pea Protein Isolate": (0.10, 0.30)
    }

    result = optimize_protein_blend(
        ingredients=plant_sources, 
        fao_reference=fao_ref, 
        bounds_config=custom_bounds,
        min_leucine=78.0
    )

    print(f"Status: {result['status']}")
    print(f"AAS: {result['score']}")
    for k, v in result['formulation'].items():
        print(f"{k}: {v}%")