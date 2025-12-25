from deepagents import create_deep_agent
from src.utils.openrouter import get_chat_model
from src.tools import WebSearchTool, YoutubeSearchTool, SaveTrendsTool
from src.config import Config
from .prompts import TREND_SCOUT_SYSTEM_PROMPT

def get_trend_agent():
    # 1. Load Model Configs from Central Config
    free_model_id = Config.LLM_FREE_MODEL
    paid_model_id = Config.LLM_PAID_MODEL
    
    print(f"   [Deep Agent] Free Model: {free_model_id}")
    print(f"   [Deep Agent] Paid Fallback: {paid_model_id}")

    # 2. Initialize Models
    llm_free = get_chat_model(model_name=free_model_id, temperature=0)
    llm_paid = get_chat_model(model_name=paid_model_id, temperature=0)

    # 3. Bind Tools Manually (Required for Fallback)
    tools = [WebSearchTool(), YoutubeSearchTool(), SaveTrendsTool()]
    llm_free_bound = llm_free.bind_tools(tools)
    llm_paid_bound = llm_paid.bind_tools(tools)

    # 4. Create Fallback Chain
    # If llm_free fails (e.g. Rate Limit), llm_paid will be tried.
    llm_with_fallback = llm_free_bound.with_fallbacks([llm_paid_bound])

    # 5. Create Agent
    # Note: We pass the fallback-enabled runnable as 'model'.
    # We still pass 'tools' so the agent framework can inject tool schemas into the prompt,
    # even though we bound them to the model already.
    agent = create_deep_agent(
        model=llm_with_fallback,
        tools=tools,
        system_prompt=TREND_SCOUT_SYSTEM_PROMPT
    )
    
    return agent
