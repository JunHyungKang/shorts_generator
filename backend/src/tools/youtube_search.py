import json
from typing import Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from youtubesearchpython import VideosSearch

# --- Input Schemas ---

class YoutubeSearchInput(BaseModel):
    keyword: str = Field(description="The keyword to search for on YouTube.")
    max_videos: int = Field(default=5, description="Maximum number of videos to retrieve.")

# --- Tool Classes ---

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
