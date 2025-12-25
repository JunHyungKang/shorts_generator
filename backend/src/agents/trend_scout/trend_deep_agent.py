from deepagents import create_deep_agent
from langchain_ollama import ChatOllama
from src.tools.trend_tools import WebSearchTool, YoutubeSearchTool
from src.tools.storage_tools import SaveTrendsTool

def get_trend_agent():
    # 1. Initialize Local Model
    # User specified 'ministral-3:3b'. 
    # Reduced context window to 4096 for stability.
    llm = ChatOllama(
        model="ministral-3:3b", 
        temperature=0,
        num_ctx=4096 
    )
    print(f"   [Deep Agent] Model: Local Ollama (ministral-3:3b) | Context: 4096")


    
    # 2. System Prompt
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
    5. **Archiving**: Finally, extract the top 3-5 confirmed trending keywords/topics and use `save_trends_tool` to save them to the database.
        
    Write your final response in Korean.
    """
    
    # 3. Create Agent with Tool Instances
    agent = create_deep_agent(
        model=llm,
        tools=[WebSearchTool(), YoutubeSearchTool(), SaveTrendsTool()],
        system_prompt=system_prompt
    )
    
    return agent
