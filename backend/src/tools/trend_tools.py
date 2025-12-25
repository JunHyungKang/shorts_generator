import json
from typing import Type, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from duckduckgo_search import DDGS
from youtubesearchpython import VideosSearch

# --- Input Schemas ---

class WebSearchInput(BaseModel):
    query: str = Field(description="The search query to find trends, news, or general information.")
    max_results: int = Field(default=3, description="Maximum number of search results to return.")

class YoutubeSearchInput(BaseModel):
    keyword: str = Field(description="The keyword to search for on YouTube.")
    max_videos: int = Field(default=5, description="Maximum number of videos to retrieve.")


# --- Tool Classes ---

class WebSearchTool(BaseTool):
    name: str = "web_search_tool"
    description: str = (
        "Search the web for a given query to find recent trends, news, or general information. "
        "Useful for discovering what is currently popular among a specific demographic."
    )
    args_schema: Type[BaseModel] = WebSearchInput

    def _run(self, query: str, max_results: int = 3) -> str:
        print(f"   [Tool] Searching Web: {query}")
        try:
            ddgs = DDGS()
            results = ddgs.text(query, max_results=max_results)
            if not results:
                return "No results found."
            
            # Format results as a string
            formatted = ""
            for i, r in enumerate(results):
                formatted += f"Result {i+1}:\nTitle: {r['title']}\nBody: {r['body']}\nSource: {r['href']}\n\n"
            return formatted
        except Exception as e:
            return f"Error during web search: {e}"

class YoutubeSearchTool(BaseTool):
    name: str = "youtube_search_tool"
    description: str = (
        "Search YouTube for videos matching a specific keyword. "
        "Returns video metadata including title, view count, and channel name. "
        "Useful for verifying if a keyword is actually popular on YouTube."
    )
    args_schema: Type[BaseModel] = YoutubeSearchInput

    def _run(self, keyword: str, max_videos: int = 5) -> str:
        print(f"   [Tool] Searching YouTube: {keyword}")
        try:
            videos_search = VideosSearch(keyword, limit=max_videos)
            results = videos_search.result()
            
            if 'result' not in results:
                return "No videos found."

            video_list = []
            for v in results['result']:
                video_data = {
                    "title": v.get('title'),
                    "views": v.get('viewCount', {}).get('short', 'N/A'),
                    "channel": v.get('channel', {}).get('name'),
                    "link": v.get('link')
                }
                video_list.append(video_data)
                
            return json.dumps(video_list, ensure_ascii=False, indent=2)
        except Exception as e:
            return f"Error during YouTube search: {e}"
