import os
import subprocess
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.agent_parser import parse_human_constraints
from src.optimize_formulation import (
    FAO_REFERENCE,
    load_plant_library,
    optimize_protein_blend,
)

DB_PATH = "data/formulation_engine.db"


def ensure_database_exists() -> None:
    """Auto-builds the SQLite database on fresh clones or cloud deployments."""
    if not os.path.exists(DB_PATH):
        subprocess.run([sys.executable, "setup_db.py"], check=True)


def render_results(result: dict) -> None:
    """Renders KPI metrics, formulation breakdown, and FAO radar chart."""
    if result["status"] != "Optimal":
        st.error(f"❌ {result['status']}")
        return

    metrics = result["metrics"]
    st.success(f"Optimal blend found using `{result['objective']}` mode.")

    # Top KPI Row
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Achieved DIAAS", f"{result['score']:.3f}")
    k2.metric(
        f"Scoop Size ({int(metrics['target_protein_g'])}g protein)",
        f"{metrics['scoop_size_g']} g",
    )
    k3.metric("Cost / Serving", f"${metrics['cost_per_serving']:.2f}")
    k4.metric("Calories / Serving", f"{metrics['calories_per_serving']} kcal")

    # Charts & Recipe Table
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Physical Scoop Breakdown (% Powder Weight)**")
        formulation = result["formulation"]
        powder_grams = result["powder_grams"]

        fig_pie = px.pie(
            names=list(formulation.keys()),
            values=list(formulation.values()),
            hole=0.4,
            color_discrete_sequence=px.colors.sequential.Teal,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        recipe_df = pd.DataFrame(
            [
                {
                    "Ingredient": ing,
                    "Powder Share (%)": pct,
                    "Weight per Scoop (g)": powder_grams[ing],
                }
                for ing, pct in formulation.items()
            ]
        )
        st.dataframe(recipe_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Digestible Amino Acid Profile (mg/g) vs. FAO Reference**")
        blend_profile = result["blend_profile"]

        categories = list(FAO_REFERENCE.keys())
        fao_values = [FAO_REFERENCE[aa] for aa in categories]
        blend_values = [blend_profile[aa] for aa in categories]

        # Close the radar polygon
        categories.append(categories[0])
        fao_values.append(fao_values[0])
        blend_values.append(blend_values[0])

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=fao_values,
                theta=categories,
                fill="toself",
                name="FAO Minimum",
                line_color="rgba(255, 99, 132, 1)",
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=blend_values,
                theta=categories,
                fill="toself",
                name="Optimized Blend",
                line_color="rgba(54, 162, 235, 1)",
            )
        )
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True)),
            showlegend=True,
        )
        st.plotly_chart(fig_radar, use_container_width=True)


# --- App Configuration & Startup ---
st.set_page_config(page_title="DIAAS Protein Formulation Engine", layout="wide")
ensure_database_exists()
all_ingredients = list(load_plant_library(DB_PATH).keys())

st.title("🧬 Prescriptive AI: Inverse Plant Protein Formulation")
st.markdown(
    "Translate dietary and commercial constraints into mathematically optimal, "
    "complete-protein plant blends using SQLite, Pydantic-validated LLMs, and SciPy Linear Programming."
)

# --- Shared Sidebar Settings ---
st.sidebar.header("Serving & Quality Targets")
target_protein_g = st.sidebar.slider(
    "Target Protein per Serving (g)", min_value=15.0, max_value=40.0, value=25.0, step=1.0
)
min_diaas = st.sidebar.slider(
    "Minimum DIAAS Floor (Cost & Calorie modes)",
    min_value=0.70,
    max_value=1.15,
    value=1.00,
    step=0.05,
)

# --- Two Modes: AI Natural Language vs. Direct Lab Controls ---
tab_ai, tab_manual = st.tabs(["🤖 AI Constraint Parser (Local Llama 3.2)", "🎛️ Manual Lab Controls"])

with tab_ai:
    user_input = st.text_area(
        "Describe your product requirements in plain English:",
        value="Build me the cheapest soy-free and gluten-free blend, and keep Hemp Seed under 20% because of the taste.",
        height=100,
    )

    if st.button("Parse & Optimize", type="primary"):
        try:
            with st.spinner("Llama 3.2 is extracting structured constraints..."):
                parsed = parse_human_constraints(user_input, all_ingredients)
        except Exception as exc:
            st.warning(
                "Local Ollama server (`llama3.2`) is unreachable. "
                "Start Ollama locally to use AI parsing, or switch to the **Manual Lab Controls** tab."
            )
            st.caption(f"Connection detail: {exc}")
        else:
            with st.expander("Validated Pydantic -> Solver Payload", expanded=True):
                st.json(parsed)

            with st.spinner("Running SQLite allergen filter & SciPy LP solver..."):
                filtered_library = load_plant_library(
                    DB_PATH, excluded_allergens=parsed["excluded_allergens"]
                )
                result = optimize_protein_blend(
                    ingredients=filtered_library,
                    objective=parsed["objective"],
                    min_diaas=min_diaas,
                    bounds_config=parsed["bounds_config"],
                    min_leucine=parsed["min_leucine"],
                    target_protein_g=target_protein_g,
                )
            render_results(result)

with tab_manual:
    objective_labels = {
        "Maximize Protein Quality (DIAAS)": "max_diaas",
        "Minimize Cost per Serving": "min_cost",
        "Minimize Calories per Serving": "min_calories",
    }

    c1, c2 = st.columns(2)
    with c1:
        selected_label = st.selectbox("Optimization Objective", list(objective_labels.keys()))
        objective = objective_labels[selected_label]
        excluded_allergens = st.multiselect("Exclude Allergens (SQL Layer)", ["Soy", "Gluten"])
    with c2:
        excluded_ingredients = st.multiselect("Exclude Specific Ingredients (0% Cap)", all_ingredients)
        enforce_leucine = st.checkbox("Enforce Minimum Digestible Leucine (mg/g protein)")
        min_leucine = (
            st.number_input("Minimum Leucine (mg/g)", min_value=40.0, max_value=110.0, value=70.0, step=1.0)
            if enforce_leucine
            else None
        )

    if st.button("Run Linear Solver", type="primary"):
        bounds_config = {ing: (0.0, 0.0) for ing in excluded_ingredients}
        with st.spinner("Running SQLite allergen filter & SciPy LP solver..."):
            filtered_library = load_plant_library(DB_PATH, excluded_allergens=excluded_allergens)
            result = optimize_protein_blend(
                ingredients=filtered_library,
                objective=objective,
                min_diaas=min_diaas,
                bounds_config=bounds_config,
                min_leucine=min_leucine,
                target_protein_g=target_protein_g,
            )
        render_results(result)