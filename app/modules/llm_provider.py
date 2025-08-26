from typing import Optional
import os
from langchain_groq import ChatGroq

ALLOWED_MODELS = {
    "llama-3-3-70b-instruct": "llama-3-3-70b-instruct",
    "gemma2-9b-it": "Gemma2-9b-It",  # allow lowercase key, preserve the exact string to pass through
}

DEFAULT_MODEL = "llama-3-3-70b-instruct"

def _normalize_model_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    return name.strip().lower()

def make_llm(groq_api_key: str, model: str = DEFAULT_MODEL, temperature: float = 0.2):
    """
    Return a LangChain ChatGroq LLM instance.

    Args:
        groq_api_key: User-provided API key (not persisted).
        model: Requested model name (must be one of ALLOWED_MODELS).
        temperature: Creativity level.
    """
    if not groq_api_key:
        raise ValueError("Missing Groq API key.")

    norm = _normalize_model_name(model)
    if norm not in ALLOWED_MODELS:
        allowed = ", ".join(ALLOWED_MODELS.keys())
        raise ValueError(
            f"Invalid model '{model}'. Allowed models: {allowed}. "
            f"Use one of the exact names (case-insensitive)."
        )

    
    canonical_model_name = ALLOWED_MODELS[norm]

    os.environ['GROQ_API_KEY'] = groq_api_key

    llm = ChatGroq(model=canonical_model_name, temperature=temperature)
    return llm
