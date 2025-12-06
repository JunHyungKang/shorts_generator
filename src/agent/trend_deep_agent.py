from langchain.agents.middleware import ModelFallbackMiddleware
from deepagents import create_deep_agent
from src.utils.openrouter import get_chat_model
from src.tools.trend_tools import web_search_tool, youtube_search_tool

def get_trend_agent():
    # 1. Define Models
    # Primary: Latest Gemini Flash (Fast & Smart)
    primary_llm = get_chat_model(model_name="google/gemini-2.0-flash-exp:free", temperature=0)
    
    # Fallback 1: Llama 3 (Good instruction following)
    fallback_1 = get_chat_model(model_name="meta-llama/llama-3.2-3b-instruct:free", temperature=0)
    
    # Fallback 2: Mistral 7B (Solid general purpose)
    fallback_2 = get_chat_model(model_name="mistralai/mistral-7b-instruct:free", temperature=0)
    
    # 2. Configure Middleware
    fallback_middleware = ModelFallbackMiddleware(fallback_1, fallback_2)
    
    # 3. System Prompt
    system_prompt = """
    You are an expert 'Senior Trend Analyst' for a content creation team.
    Your goal is to identify high-potential YouTube video topics for the Korean senior demographic (60s-70s).
    
    You must follow this workflow:
    1. **Discovery**: Use `web_search_tool` to find recent interests, news, or trends relevant to Korean seniors (e.g., health, finance, hobbies, trot music).
    2. **Hypothesis**: Based on the research, formulate 3-5 specific YouTube search keywords that might be popular.
    3. **Verification**: Use `youtube_search_tool` to search for these keywords and check actual video performance (views).
    4. **Report**: Synthesize your findings into a final report summarizing:
        - What topic is hot currently?
        - What kind of video titles/thumbnails are working?
        - Recommendations for a new video.
        
    Write your final response in Korean.
    """
    
    # 4. Create Agent
    agent = create_deep_agent(
        model=primary_llm,
        tools=[web_search_tool, youtube_search_tool],
        middleware=[fallback_middleware],
        system_prompt=system_prompt
    )
    
    return agent
