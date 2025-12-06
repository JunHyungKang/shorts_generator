import os
import requests
from typing import List
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def fetch_free_models() -> List[str]:
    """
    Fetch all currently free models from OpenRouter API.
    Returns a list of model IDs.
    """
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        if response.status_code == 200:
            data = response.json().get("data", [])
            # Filter for models where pricing is explicitly 0
            free_models = [
                m["id"] for m in data 
                if m.get("pricing", {}).get("prompt") == "0" 
                and m.get("pricing", {}).get("completion") == "0"
            ]
            # Prioritize Google/Meta/Mistral/Microsoft for quality
            priority = []
            others = []
            for m in free_models:
                if any(k in m for k in ["google", "meta-llama", "mistral", "microsoft"]):
                    priority.append(m)
                else:
                    others.append(m)
            
            # Return sorted list (Priority first, then others)
            return priority + others
    except Exception as e:
        print(f"   [Warning] Failed to fetch free models: {e}")
        pass
    
    # Fallback list if fetch fails
    return [
        "google/gemini-2.0-flash-exp:free",
        "meta-llama/llama-3.2-3b-instruct:free",
        "mistralai/mistral-7b-instruct:free",
        "microsoft/phi-3-mini-128k-instruct:free",
        "google/gemini-exp-1206:free",
        "huggingfaceh4/zephyr-7b-beta:free",
        "openchat/openchat-7b:free",
    ]

def get_chat_model(model_name: str = None, temperature: float = 0.7, use_free_fallback: bool = True):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY could not be found in environment variables.")

    # Default to Gemini Flash Free if not specified
    if model_name is None:
        model_name = "google/gemini-2.0-flash-exp:free"

    extra_body = {}
    
    if use_free_fallback:
        # Fetch dynamic list of free models
        free_models = fetch_free_models()
        
        # Remove current model from fallback list to avoid redundancy logic issues
        fallbacks = [m for m in free_models if m != model_name]
        
        if fallbacks:
            # OpenRouter allows multiple models in 'models' field (Max 3).
            # If the primary 'model_name' fails or is busy, it routes to these.
            extra_body["models"] = fallbacks[:3]
            
            # Explicitly tell OpenRouter to load balance/route based on availability/price (free)
            # "orders" field can also be used, but "models" + "provider.sort" is effective.
            extra_body["provider"] = {
                "sort": "price", # Prefer lowest price (free)
                # "allow_fallbacks": True # Default is true
            }

    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=model_name,
        temperature=temperature,
        model_kwargs={
            "extra_body": extra_body
        },
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Shorts Generator Agent"
        }
    )
    return llm
