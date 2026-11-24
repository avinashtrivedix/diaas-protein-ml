from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama


class IngredientBound(BaseModel):
    ingredient: str = Field(description = "Exact ingredient name from the available list.")
    min_share: float = Field(description= "Minimum share as a decimal between 0.0 and 1.0")
    max_share: float = Field(description= "Maximum share as a decimal between 0.0 and 1.0")

class FormulationConstraints(BaseModel):
    objective: Literal["max_diaas", "min_cost", "min_calories"] = Field(
        description=(
            "Optimization goal: 'min_cost' if user asks for cheap/budget/affordable, "
            "'min_calories' if user asks for low-calorie/lean/cutting, "
            "otherwise 'max_diaas'."
        ),
    )
    excluded_allergens: List[Literal["Soy", "Gluten"]] = Field(
        description="Allergens to exclude at the SQL layer. Include 'Soy' if soy-free, 'Gluten' if gluten-free.",
    )
    ingredient_bounds: List[IngredientBound] = Field(
        description="List of specific ingredients limits mentioned by the user. Empty list [] if none."
    )
    min_leucine: Optional[float] = Field(
        description="Minimum required digestible Leucine in mg/g protein, or null if not mentioned.",
    )


def _sanitize_bounds(
    raw_bounds: List[IngredientBound], available_ingredients: List[str]
) -> Dict[str, Tuple[float, float]]:
    """Guards the LP solver against LLM case mismatches, hallucinated names, or inverted bounds."""
    lookup = {name.lower(): name for name in available_ingredients}
    clean_bounds: Dict[str, Tuple[float, float]] = {}

    for item in raw_bounds:
        matched_name = lookup.get(item.ingredient.strip().lower())
        if not matched_name:
            continue

        low = max(0.0, min(1.0, float(item.min_share)))
        high = max(0.0, min(1.0, float(item.max_share)))

        if low <= high:
            clean_bounds[matched_name] = (low, high)

    return clean_bounds


def parse_human_constraints(user_input: str, available_ingredients: List[str]) -> dict:
    """Translates plain-English formulation requests into validated LP constraints via Llama 3.2."""
    llm = ChatOllama(model="llama3.2", temperature=0)
    structured_llm = llm.with_structured_output(FormulationConstraints, method="json_schema")

    prompt = f"""
You are a food science optimization assistant extracting strict mathematical constraints.
Available ingredient names: {', '.join(available_ingredients)}

Rules:
1. objective: 'min_cost' (cheap/budget), 'min_calories' (low-calorie/lean), or 'max_diaas' (default).
2. excluded_allergens: include 'Soy' for soy-free/no soy, 'Gluten' for gluten-free/no wheat. Otherwise [].
3. ingredient_bounds: ONLY include ingredients explicitly named in the User Request. NEVER include unmentioned ingredients.
   - "no X" -> min_share: 0.0, max_share: 0.0
   - "under 20% X" -> min_share: 0.0, max_share: 0.20
   - "at least 30% X" -> min_share: 0.30, max_share: 1.0
4. min_leucine: float in mg/g if mentioned, otherwise null.

Example Input: "Make a low-calorie gluten-free blend with at least 30% Pea Isolate and no Chia Seed."
Example Output:
{{
  "objective": "min_calories",
  "excluded_allergens": ["Gluten"],
  "ingredient_bounds": [
    {{"ingredient": "Pea Isolate", "min_share": 0.3, "max_share": 1.0}},
    {{"ingredient": "Chia Seed", "min_share": 0.0, "max_share": 0.0}}
  ],
  "min_leucine": null
}}

User Request: "{user_input}"
"""
    result = structured_llm.invoke(prompt)
    return {
        "objective": result.objective,
        "excluded_allergens": result.excluded_allergens,
        "bounds_config": _sanitize_bounds(result.ingredient_bounds, available_ingredients),
        "min_leucine": result.min_leucine,
    }


if __name__ == "__main__":
    from src.optimize_formulation import load_plant_library

    library = load_plant_library()
    ingredients_list = list(library.keys())

    test_prompt = (
        "Build me the cheapest soy-free and gluten-free blend, "
        "and keep Hemp Seed under 20% because of the taste."
    )

    print(f"Human Request: '{test_prompt}'\n")
    parsed = parse_human_constraints(test_prompt, ingredients_list)
    print("Validated Constraints:")
    print(parsed)