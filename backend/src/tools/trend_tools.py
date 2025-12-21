import json
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from youtubesearchpython import VideosSearch

@tool
def web_search_tool(query: str, max_results: int = 3) -> str:
    """
    Search the web for a given query to find recent trends, news, or general information.
    Useful for discovering what is currently popular among a specific demographic.
    """
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

@tool
def youtube_search_tool(keyword: str, max_videos: int = 5) -> str:
    """
    Search YouTube for videos matching a specific keyword. 
    Returns video metadata including title, view count, and channel name.
    Useful for verifying if a keyword is actually popular on YouTube.
    """
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
