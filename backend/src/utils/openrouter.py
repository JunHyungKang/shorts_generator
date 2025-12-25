from src.config import Config

def get_chat_model(model_name: str = None, temperature: float = 0.7):
    api_key = Config.OPENROUTER_API_KEY
    
    # Default to Free Model from Config if not specified
    if model_name is None:
        model_name = Config.LLM_FREE_MODEL

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
