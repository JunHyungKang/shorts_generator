import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Load environment variables (ensure .env is loaded if called from anywhere)
load_dotenv()

FREE_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "meta-llama/llama-3.2-11b-vision-instruct:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "mistralai/mistral-7b-instruct:free",
    "microsoft/phi-3-mini-128k-instruct:free",
]

def get_chat_model(model_name: str = None, temperature: float = 0.7, use_free_fallback: bool = True):
    """
    Returns a ChatOpenAI instance configured for OpenRouter.
    
    Args:
        model_name (str): The primary model ID to use. If None, defaults to the first model in FREE_MODELS.
        temperature (float): The sampling temperature.
        use_free_fallback (bool): If True, automatically adds other free models as fallbacks.
        
    Returns:
        ChatOpenAI: Configured LangChain chat model.
        
    Raises:
        ValueError: If OPENROUTER_API_KEY is not set in environment.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY could not be found in environment variables.")

    if model_name is None:
        model_name = FREE_MODELS[0]

    # Configure fallback models
    extra_body = {}
    if use_free_fallback:
        # Create a list of fallbacks excluding the primary model
        fallbacks = [m for m in FREE_MODELS if m != model_name]
        if fallbacks:
            # OpenRouter limits 'models' array to 3 items or fewer
            extra_body["models"] = fallbacks[:3]

    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=model_name,
        temperature=temperature,
        model_kwargs={
            "extra_body": extra_body
        },
        default_headers={
            "HTTP-Referer": "https://localhost:3000",
            "X-Title": "Shorts Generator"
        }
    )
    return llm
