import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Protein Quality Predictor", layout="wide")

# Load model artifact
@st.cache_resource
def load_artifact():
    artifact_path = Path("models/protein_rf_model.joblib")
    if not artifact_path.exists():
        artifact_path = Path("../models/protein_rf_model.joblib")
    return joblib.load(artifact_path)

data = load_artifact()
model = data["model"]
feature_cols = data["feature_cols"]
fao_ref = data["fao_reference"]

st.title("Protein Quality & Amino Acid Score Engine")
st.markdown("Predict amino acid limiting scores and DIAAS protein quality benchmarks from essential amino acid profiles (mg/g protein).")

# Preset selection for rapid testing
presets = {
    "Custom": {},
    "Egg Whole": {"His": 22.0, "Ile": 54.0, "Leu": 86.0, "Lys": 70.0, "Met": 30.0, "Cys": 23.0, "Phe": 53.0, "Tyr": 40.0, "Thr": 47.0, "Trp": 16.0, "Val": 66.0},
    "Soy Flour": {"His": 26.0, "Ile": 48.0, "Leu": 78.0, "Lys": 63.0, "Met": 13.0, "Cys": 14.0, "Phe": 50.0, "Tyr": 37.0, "Thr": 39.0, "Trp": 13.0, "Val": 49.0},
    "Wheat Grain": {"His": 23.0, "Ile": 36.0, "Leu": 67.0, "Lys": 28.0, "Met": 16.0, "Cys": 23.0, "Phe": 48.0, "Tyr": 31.0, "Thr": 28.0, "Trp": 12.0, "Val": 43.0}
}

selected_preset = st.selectbox("Load Example Amino Acid Profile", list(presets.keys()))

st.subheader("Amino Acid Concentrations (mg / g crude protein)")
cols = st.columns(4)

inputs = {}
amino_list = ["His", "Ile", "Leu", "Lys", "Met", "Cys", "Phe", "Tyr", "Thr", "Trp", "Val"]

for idx, aa in enumerate(amino_list):
    col = cols[idx % 4]
    default_val = presets[selected_preset].get(aa, 30.0) if selected_preset != "Custom" else 30.0
    inputs[aa] = col.number_input(f"{aa} (FAO Ref: {fao_ref.get(aa, 'N/A')})", min_value=1.0, max_value=250.0, value=float(default_val), step=1.0)

# Calculate SAA and AAA
inputs["SAA"] = inputs["Met"] + inputs["Cys"]
inputs["AAA"] = inputs["Phe"] + inputs["Tyr"]

if st.button("Evaluate Quality", type="primary"):
    # 1. Deterministic FAO scoring
    ratios = {aa: inputs[aa] / fao_ref[aa] for aa in fao_ref}
    limiting_aa = min(ratios, key=ratios.get)
    exact_score = ratios[limiting_aa]

    # 2. Compositional CLR feature transformation
    raw_vector = np.array([inputs[col.replace("clr_", "")] for col in feature_cols])
    geo_mean = np.exp(np.mean(np.log(raw_vector)))
    clr_vals = np.log(raw_vector / geo_mean).reshape(1, -1)
    clr_df = pd.DataFrame(clr_vals, columns=feature_cols)

    # 3. Predict via Random Forest
    pred_score = model.predict(clr_df)[0]

    # Display results
    st.divider()
    res_col1, res_col2, res_col3 = st.columns(3)
    
    res_col1.metric("Limiting Amino Acid", limiting_aa)
    res_col2.metric("Exact AAS (FAO 2013)", f"{exact_score:.2f}")
    res_col3.metric("RF Model Prediction", f"{pred_score:.2f}")

    if exact_score >= 1.0:
        st.success("Status: Complete Protein (Meets or exceeds 100% of human reference requirements).")
    else:
        st.warning(f"Status: Incomplete Protein (Deficient in {limiting_aa}; supplies only {exact_score*100:.1f}% of body requirement).")

    # Amino acid breakdown chart
    st.subheader("Amino Acid Sufficiency vs. FAO Requirement (1.0 = 100%)")
    df_chart = pd.DataFrame.from_dict(ratios, orient="index", columns=["Ratio"])
    st.bar_chart(df_chart)