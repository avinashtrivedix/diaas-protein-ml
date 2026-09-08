import streamlit as st
from src.optimize_formulation import optimize_protein_blend
from src.agent_parser import parse_human_contraints_llama

# --- 1. Define the Core Data ---
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

# --- 2. Build the UI ---
st.set_page_config(page_title="AI Formulation Agent", layout="wide")
st.title("🧬 Prescriptive AI: Inverse Protein Formulation")
st.markdown("Enter your product requirements in plain English. The LLM will parse your constraints and the Linear Programming solver will find the mathematically optimal blend.")

# User Input
user_input = st.text_area(
    "Formulation Constraints:", 
    value="I need a sports blend with at least 85mg of leucine. Make it totally soy-free, and keep the hemp seed under 20%."
)

if st.button("Generate Optimal Blend"):
    with st.spinner("LLM is extracting math constraints..."):
        # Step A: Brain (LLM)
        bounds, min_leucine = parse_human_contraints_llama(user_input, list(plant_sources.keys()))
        
        st.success("Constraints Parsed Successfully!")
        st.json({"Extracted_Bounds": bounds, "Target_Leucine": min_leucine})

    with st.spinner("SciPy is running linear optimization..."):
        # Step B: Muscle (SciPy)
        result = optimize_protein_blend(
            ingredients=plant_sources, 
            fao_reference=fao_ref, 
            bounds_config=bounds,
            min_leucine=min_leucine
        )
        
        # Step C: Visualization
        if result["status"] == "Optimal":
            st.subheader(f"✅ Optimal Formulation Found (AAS: {result['score']})")
            
            # Display metrics
            cols = st.columns(len(result["formulation"]))
            for col, (ingredient, percentage) in zip(cols, result["formulation"].items()):
                col.metric(label=ingredient, value=f"{percentage}%")
        else:
            st.error("❌ Infeasible: No mathematical combination of these proteins can satisfy your constraints.")