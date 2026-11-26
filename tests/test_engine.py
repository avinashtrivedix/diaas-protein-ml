import pytest

from src.agent_parser import IngredientBound, _sanitize_bounds
from src.optimize_formulation import load_plant_library, optimize_protein_blend


def test_sql_allergen_excludes_soy():
    library = load_plant_library(excluded_allergens = ["Soy"])
    assert len(library) > 0
    for name, props in library.items():
        assert props["Allergen"] != "Soy", f"Soy leaked through filter: {name}"

def test_sanitize_bounds_blocks_bad_llm_outouts():
    available = ["Hemp Seed", "Pea Protein"]
    messy_llm_output = [
        IngredientBound(ingredient = " hemp seed ", min_share= 0.0, max_share = 0.2),
        IngredientBound(ingredient = "Unicorn Dust", min_share=0.0, max_share=0.5),
        IngredientBound(ingredient = "Pea Isolate", min_share=0.8, max_share=0.2),
    ]
    cleaned = _sanitize_bounds(messy_llm_output, available)
    assert cleaned == {"Hemp Seed" : (0.0, 0.2)}

def test_impossible_diaas_returns_infeasible():
    library = load_plant_library()
    res = optimize_protein_blend(library, objective = "min_cost", min_diaas  = 1.5)
    assert "Infeasible" in res["status"]

def test_min_cost_is_not_more_expensive_than_max_diaas():
    library = load_plant_library()
    res_max_diaas = optimize_protein_blend(library, objective = "max_diaas")
    res_cost = optimize_protein_blend(library, objective= "min_cost", min_diaas = 1.0)
    assert res_cost["metrics"]["cost_per_serving"] <= res_max_diaas["metrics"]["cost_per_serving"]

def test_scoop_percentage_sum_to_100():
    library = load_plant_library()
    res = optimize_protein_blend(library, objective= "max_diaas")
    assert pytest.approx(100.0, abs=0.2) == sum(res["formulation"].values())
