import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.optimize_formulation import optimize_protein_blend
from src.agent_parser import parse_human_contraints_llama

# --- 1. Define the Core Data ---
fao_ref = {
    "His": 16.0, "Ile": 30.0, "Leu": 59.0, "Lys": 45.0,
    "SAA": 22.0, "AAA": 38.0, "Thr": 23.0, "Trp": 6.0, "Val": 39.0
}

plant_sources = {
    "Pea Protein Isolate": {"digestibility": 0.89, "His": 25.0, "Ile": 43.0, "Leu": 66.0, "Lys": 72.0, "SAA": 19.0, "AAA": 86.0, "Thr": 38.0, "Trp": 10.0, "Val": 50.0},
    "Brown Rice Protein": {"digestibility": 0.85, "His": 23.0, "Ile": 41.0, "Leu": 82.0, "Lys": 31.0, "SAA": 38.0, "AAA": 85.0, "Thr": 37.0, "Trp": 11.0, "Val": 58.0},
    "Soy Protein Isolate": {"digestibility": 0.90, "His": 26.0, "Ile": 49.0, "Leu": 82.0, "Lys": 63.0, "SAA": 26.0, "AAA": 90.0, "Thr": 38.0, "Trp": 13.0, "Val": 50.0},
    "Hemp Seed Protein": {"digestibility": 0.84, "His": 28.0, "Ile": 38.0, "Leu": 66.0, "Lys": 38.0, "SAA": 41.0, "AAA": 80.0, "Thr": 34.0, "Trp": 12.0, "Val": 52.0}
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
        # st.json({"Extracted_Bounds": bounds, "Target_Leucine": min_leucine})

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
            st.subheader(f"✅ Optimal Formulation Found (DIAAS: {result['score']})")
            
            formulation_data = result['formulation']
            active_ingredients = {k: v for k, v in formulation_data.items() if v > 0}

            # --- PREPARE THE RADAR CHART DATA ---
            # 1. Calculate the exact amino acid profile of the final blend
            blend_profile = {}
            for aa in fao_ref.keys():
                total_aa = 0
                for ing, pct in active_ingredients.items():
                    weight = pct / 100.0
                    digestible_aa = plant_sources[ing][aa] * plant_sources[ing].get("digestibility", 1.0)
                    total_aa += weight * digestible_aa
                blend_profile[aa] = total_aa

            # 2. Setup plotting lists (Plotly needs the first item duplicated at the end to close the circle)
            categories = list(fao_ref.keys())
            fao_values = list(fao_ref.values())
            blend_values = [blend_profile[aa] for aa in categories]

            categories.append(categories[0])
            fao_values.append(fao_values[0])
            blend_values.append(blend_values[0])

            # --- BUILD THE VISUAL DASHBOARD ---
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Formulation Breakdown**")
                fig_pie = px.pie(
                    names=list(active_ingredients.keys()),
                    values=list(active_ingredients.values()),
                    hole=0.4, 
                    color_discrete_sequence=px.colors.sequential.Teal
                )
                st.plotly_chart(fig_pie, use_container_width=True)

            with col2:
                st.markdown("**Amino Acid Profile (mg/g) vs FAO Requirement**")
                fig_radar = go.Figure()
                
                # Draw the FAO Baseline Polygon
                fig_radar.add_trace(go.Scatterpolar(
                    r=fao_values,
                    theta=categories,
                    fill='toself',
                    name='FAO Minimum',
                    line_color='rgba(255, 99, 132, 1)'
                ))
                
                # Draw Your Custom Blend Polygon on Top
                fig_radar.add_trace(go.Scatterpolar(
                    r=blend_values,
                    theta=categories,
                    fill='toself',
                    name='Optimized Blend',
                    line_color='rgba(54, 162, 235, 1)'
                ))
                
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
                st.plotly_chart(fig_radar, use_container_width=True)
                
    else:
        st.error("❌ Infeasible: No mathematical combination of these proteins can satisfy your constraints.")