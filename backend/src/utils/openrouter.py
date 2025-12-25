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
                m for m in data 
                if m.get("pricing", {}).get("prompt") == "0" 
                and m.get("pricing", {}).get("completion") == "0"
            ]
            
            # Sort by 'created' timestamp descending (Newest first)
            free_models.sort(key=lambda x: x.get("created", 0), reverse=True)
            
            return [m["id"] for m in free_models]
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

def get_model_candidates(temperature: float = 0.7) -> List[ChatOpenAI]:
    """
    Returns a list of ChatOpenAI instances for ALL currently free models.
    Useful for client-side fallback chains (Middleware).
    """
    model_ids = fetch_free_models()
    candidates = []
    
    for model_id in model_ids:
        try:
            llm = get_chat_model(model_name=model_id, temperature=temperature)
            candidates.append(llm)
        except Exception as e:
            print(f"   [Warning] Failed to initialize candidate model {model_id}: {e}")
            continue
        
    return candidates

def get_chat_model(model_name: str = None, temperature: float = 0.7, use_free_fallback: bool = True):
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY could not be found in environment variables.")

    # Default to Gemini Flash Free if not specified
    if model_name is None:
        model_name = "google/gemini-2.0-flash-exp:free"

    # Simplified: No complex server-side fallback instructions here.
    # We rely on the client-side Middleware to handle rotation if this simple instance fails.
    
    llm = ChatOpenAI(
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
        model_name=model_name,
        temperature=temperature,
        default_headers={
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "Shorts Generator Agent"
        }
    )
    return llm
