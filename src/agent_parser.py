from typing import Dict, Tuple, Optional
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama

# 1. Define the exact strict schema the AI MUST follow
class FormulationConstraints(BaseModel):
    bounds_config: Dict[str, Tuple[float, float]] = Field(
        description="Dictionary of ingredient names mapped to (min, max) allowed percentages. 0.0 means 0%, 1.0 means 100%."
    )
    min_leucine: Optional[float] = Field(
        description="The minimum required mg of leucine, or null if not mentioned."
    )

def parse_human_contraints_llama(user_input, available_ingredients):
    """
    Uses a local LLaMA model with Pydantic Structured Outputs to guarantee exact math formatting.
    """
    # 2. Use the modern Chat model, set temperature to 0 for strict logic
    llm = ChatOllama(model="llama3.2", temperature=0)

    # 3. Lock the LLM to our Pydantic schema
    structured_llm = llm.with_structured_output(FormulationConstraints)

    # Notice how much simpler the prompt is now that Pydantic handles the formatting
    prompt = f"""
    You are an AI formulation scientist extracting mathematical constraints.
    Available Ingredients: {', '.join(available_ingredients)}
    
    Rules:
    - If user says "no [ingredient]", "zero", or "free", set bounds to [0.0, 0.0]
    - If user sets a cap (e.g., "under 20%"), set bounds to [0.0, 0.20]
    - If an ingredient is NOT mentioned, you MUST set its bounds to [0.0, 1.0]
    
    User Request: "{user_input}"
    """
    
    print("🧠 LLM is reasoning with Structured Outputs...")
    
    # 4. Invoke returns a perfect Pydantic object, not a messy string
    result = structured_llm.invoke(prompt)
    
    # Extract the validated Python properties
    return result.bounds_config, result.min_leucine

# ==========================================
# TEST THE AGENT
# ==========================================
if __name__ == "__main__":
    ingredients_list = [
        "Pea Protein Isolate", "Brown Rice Protein", 
        "Soy Protein Isolate", "Hemp Seed Protein"
    ]
    
    test_prompt = "I need a sports blend with at least 85mg of leucine. Make it totally soy-free, and keep the hemp seed under 20% because of the taste."
    
    print(f"Human Request: '{test_prompt}'\n")
    
    bounds, leu = parse_human_contraints_llama(test_prompt, ingredients_list)
    
    print("✅ Pydantic Validated Matrix Constraints:")
    print(f"Bounds Config: {bounds}")
    print(f"Min Leucine: {leu}")