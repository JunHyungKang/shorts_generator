import os
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from tavily import TavilyClient

# --- Input Schemas ---

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to find trends, news, or general information.")
    max_results: int = Field(default=3, description="Maximum number of search results to return.")

# --- Tool Classes ---

class WebSearchTool(BaseTool):
    name: str = "web_search_tool"
    description: str = (
        "Search the web for a given query to find recent trends, news, or general information using Tavily. "
        "Useful for discovering what is currently popular among a specific demographic."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str, max_results: int = 3) -> str:
        print(f"   [Tool] Searching Web (Tavily): {query}")
        
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Error: TAVILY_API_KEY is not set in the environment."

        try:
            client = TavilyClient(api_key=api_key)
            # Tavily 'search' returns a dict with 'results' list
            response = client.search(query=query, max_results=max_results)
            results = response.get("results", [])
            
            if not results:
                return "No results found."
            
            # Format results as a string
            formatted = ""
            for i, r in enumerate(results):
                title = r.get('title', 'No Title')
                content = r.get('content', 'No Content')
                url = r.get('url', 'No URL')
                formatted += f"Result {i+1}:\nTitle: {title}\nBody: {content}\nSource: {url}\n\n"
            return formatted
        except Exception as e:
            return f"Error during web search: {e}"
