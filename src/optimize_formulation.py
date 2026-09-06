import numpy as np
import pandas as pd
from scipy.optimize import linprog


# 1. FAO/WHO 2013 Adult Reference Pattern (mg / g protein)
fao_reference = {
    "His": 16.0, "Ile": 30.0, "Leu": 59.0, "Lys": 45.0,
    "SAA": 22.0, "AAA": 38.0, "Thr": 23.0, "Trp": 6.0, "Val": 39.0
}
aa_keys = list(fao_reference.keys())

# 2. Plant Protein Profiles (mg / g protein) - Standard Commercial Isolates
# Notice the complementary flaws: Pea is low in SAA; Rice is low in Lys.
ingredients = {
    "Pea Protein Isolate": {
        "His": 25.0, "Ile": 43.0, "Leu": 66.0, "Lys": 72.0,
        "SAA": 19.0, "AAA": 86.0, "Thr": 38.0, "Trp": 10.0, "Val": 50.0
    },
    "Brown Rice Protein": {
        "His": 23.0, "Ile": 41.0, "Leu": 82.0, "Lys": 31.0,
        "SAA": 38.0, "AAA": 85.0, "Thr": 37.0, "Trp": 11.0, "Val": 58.0
    },
    "Soy Protein Isolate": {
        "His": 26.0, "Ile": 49.0, "Leu": 82.0, "Lys": 63.0,
        "SAA": 26.0, "AAA": 90.0, "Thr": 38.0, "Trp": 13.0, "Val": 50.0
    },
    "Hemp Seed Protein": {
        "His": 28.0, "Ile": 38.0, "Leu": 66.0, "Lys": 38.0,
        "SAA": 41.0, "AAA": 80.0, "Thr": 34.0, "Trp": 12.0, "Val": 52.0
    }
}


ingredient_names = list(ingredients.keys())
n_ingredients = len(ingredient_names)
n_aa = len(aa_keys)

# BUild matrix A (shape: n_ingredients x n_aa)
# Build matrix A (shape: n_ingredients x n_aa)
A = np.array([[ingredients[ing][aa] for aa in aa_keys] for ing in ingredient_names])
fao_vec = np.array([fao_reference[aa] for aa in aa_keys])

# 3. Formulate the Linear Program
# Variables: x = [w_1, w_2, ..., w_k, t] (length: k + 1)
# Objective: Minimize -t (maximize t)
c = np.zeros(n_ingredients + 1)
c[-1] = -1.0

# Inequality constraints: t - sum(w_i * (A_ij / fao_j)) <= 0
# Represented as: A_ub * x <= b_ub
A_ub = np.zeros((n_aa, n_ingredients + 1))
for j in range(n_aa):
    for i in range(n_ingredients):
        A_ub[j, i] = -A[i, j] / fao_vec[j]
    A_ub[j, -1] = 1.0  # coefficient for t

b_ub = np.zeros(n_aa)

# Equality constraint: sum(w_i) = 1, coefficient for t = 0
A_eq = np.zeros((1, n_ingredients + 1))
A_eq[0, :n_ingredients] = 1.0
A_eq[0, -1] = 0.0
b_eq = np.array([1.0])

# Bounds: w_i >= 0, t >= 0
bounds = [(0, 1) for _ in range(n_ingredients)] + [(0, None)]

# 4. Solve the LP
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

# 5. Display the Optimal Formulation
optimal_weights = res.x[:n_ingredients]
optimal_aas = res.x[-1]

print("=== OPTIMAL PLANT BLEND FORMULATION ===")
for name, weight in zip(ingredient_names, optimal_weights):
    print(f"{name:25}: {weight * 100:5.1f}%")

print(f"\nResulting Protein Quality Score (AAS): {optimal_aas:.3f}")