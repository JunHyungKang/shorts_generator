import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

    # Model Configuration
    # Defaults are set in code as requested, but can be overridden by .env
    LLM_FREE_MODEL = "xiaomi/mimo-v2-flash:free"
    LLM_PAID_MODEL = "google/gemini-3-flash-preview"

    @classmethod
    def validate(cls):
        if not cls.OPENROUTER_API_KEY:
            raise ValueError("OPENROUTER_API_KEY is missing in environment variables.")
        if not cls.TAVILY_API_KEY:
            print("[Warning] TAVILY_API_KEY is missing. Web search may fail.")

# Validate configuration on import
Config.validate()
