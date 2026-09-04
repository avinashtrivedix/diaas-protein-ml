# Protein Quality & Amino Acid Bottleneck Prediction

An end-to-end machine learning pipeline and inference engine designed to evaluate dietary protein quality from essential amino acid profiles. This project models non-linear nutritional bottlenecks (Liebig’s Law of the Minimum) across compositional simplex data using Centered Log-Ratio (CLR) transformations and tree-based ensembles.

---

## Technical Overview

Determining dietary protein quality requires assessing how well a food's amino acid composition meets human physiological requirements. Traditional evaluation metrics (such as AAS and DIAAS) are governed by the **limiting amino acid**—the single essential amino acid present in the lowest concentration relative to human requirement patterns defined by the FAO/WHO:

$$\text{AAS} = \min_{i} \left( \frac{\text{Amino Acid}_i \text{ (mg/g protein)}}{\text{FAO Reference}_i \text{ (mg/g protein)}} \right)$$

This project implements a regression pipeline that maps raw, closed-composition amino acid profiles to their true limiting score and identifies the bottleneck amino acid.

---

## Core Engineering & Mathematical Methodology

### 1. Compositional Data & Aitchison Geometry
Amino acid profiles exist on a constrained simplex ($\sum x_i = 1$). Applying standard Euclidean metrics directly to raw percentages or concentrations induces spurious negative correlations (the closure problem). 

To project the data into unconstrained Euclidean space while preserving relative proportions, we apply the **Centered Log-Ratio (CLR)** transformation:

$$\text{clr}(x_i) = \ln \left( \frac{x_i}{g(x)} \right), \quad \text{where } g(x) = \left( \prod_{j=1}^{D} x_j \right)^{1/D}$$

* $x_i$: Concentration of essential amino acid $i$ in mg/g crude protein.
* $g(x)$: Geometric mean across the essential amino acid vector.

### 2. Formulating the Ground-Truth Target ($y$)
Early iterations attempting to predict absolute protein weight ($g / 100g$) from relative compositional ratios failed due to a fundamental scale mismatch ($X$ represented relative percentages, while $y$ was unconstrained absolute mass). 

The target was reformulated to the **Amino Acid Score (AAS)** using the FAO/WHO 2013 adult baseline:
* **Target ($y$):** Bounded continuous score where $y \ge 1.0$ indicates a complete protein and $y < 1.0$ indicates an incomplete protein.
* **Distribution:** Well-conditioned ($N=47$, Mean $= 1.08$, Median $= 1.03$, Range: $[0.21, 1.66]$).

---

## Model Benchmarking & Inductive Bias

We benchmarked linear regularization against non-linear decision tree ensembles using 5-Fold Cross-Validation:

| Model Architecture | Cross-Validated $R^2$ (Mean) | 5-Fold Scores | Inductive Fit |
| :--- | :--- | :--- | :--- |
| **Ridge Regression ($\alpha=1.0$)** | **0.421** | `[0.562, 0.206, 0.263, 0.691, 0.384]` | Poor (Linear plane cannot capture dynamic $\min()$ operators) |
| **Random Forest Regressor** | **0.698** | `[0.864, 0.623, 0.498, 0.844, 0.663]` | Strong (Hierarchical splits capture bottleneck conditions) |

### Why Trees Outperform Linear Models
The target function is fundamentally non-linear:
$$\text{Score} = \min(R_{\text{Val}}, R_{\text{Lys}}, R_{\text{Leu}}, \dots)$$

* **Ridge Regression** attempts to fit a global hyper-plane across all dimensions simultaneously. Because dairy is bottlenecked by Valine while nuts and cereals are bottlenecked by Lysine, a global linear weight vector gets pulled in conflicting directions.
* **Random Forest** naturally models piecewise conditional decision boundaries (e.g., *if $\text{clr\_Lys} < \theta_1 \to$ predict via Lysine split; else check Valine split*).

---

## Out-of-Fold Residual Diagnostics & Failure Modes

An out-of-fold residual analysis (`cross_val_predict`) was conducted across the 47 high-precision experimental profiles to diagnose model limitations:
Food Description             Limiting AA   Actual AAS   Predicted AAS   Residual (y - y_hat)
Nectarines, raw              Leu           0.208        0.989           -0.781
Mustard, prepared, yellow    Trp           0.392        1.040           -0.648
Flaxseed, ground             Lys           1.149        0.816           +0.333
Sesame butter, creamy        Lys           0.758        1.080           -0.321
Nuts, pine nuts, raw         Lys           0.830        1.149           -0.318


### Identified Failure Mechanisms
1. **Zero-Shot Representation in CV Folds (Mustard / Trp):**
   * Tryptophan is the limiting amino acid in only 1 out of 47 foods (Mustard). When Mustard was held out in validation folds, the training trees had zero examples of Tryptophan-limited profiles, leading to an overprediction of $+0.65$.
2. **Leaf Node Shrinkage at Extremes (Nectarines / Leu):**
   * Random Forest regressor predictions are bounded by the mean of training targets inside terminal leaves. Because 75% of the training distribution sits between $0.84$ and $1.66$, extreme outliers ($0.21$) suffer from regression to the mean.
3. **Core Cluster Performance:**
   * For foods within standard dietary bounds ($[0.60, 1.60]$), the model reliably tracks nutritional quality with an average absolute error of $\approx 0.18$.

---

## Project Structure

```bash
├── data/
│   ├── processed/
│   │   └── usda_amino_acid_profiles.csv   # Normalized experimental profiles (mg/g)
│   └── features/
│       └── features_clr_transformed.csv   # CLR transformed feature space
├── models/
│   └── protein_rf_model.joblib            # Serialized pipeline (model + FAO reference schema)
├── notebooks/
│   └── exploratory_modeling.ipynb        # EDA, CLR transformations, CV & error analysis
├── app.py                                 # Streamlit deployment dashboard
├── requirements.txt                       # Reproducible dependency specification
└── README.md


Getting Started
1. Clone & Setup Environment
Bash
git clone [https://github.com/](https://github.com/)<your-username>/diaas-protein-ml.git
cd diaas-protein-ml

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

2. Launch the Streamlit DashboardBashstreamlit run app.py
Key Takeaways & Production RoadmapCompositional Primitives: Log-ratio transformations (CLR) are mandatory when modeling proportional biological compositions to remove simplex correlation artifacts.Non-linear Bottleneck Modeling: Ensembles of decision trees are fundamentally better suited than linear regressions for modeling Liebig bottleneck functions.Next Steps: Ingest USDA SR Legacy to expand the training cohort from $N=47$ to $N \approx 1,200$, addressing the extreme-tail sparsity observed in Tryptophan and low-protein fruits.