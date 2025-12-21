import json
import os
from typing import List
from langchain_core.tools import tool

DATA_FILE = "data/trends.json"

@tool
def save_trends_tool(new_trends: List[str]) -> str:
    """
    Save a list of confirmed trending keywords/topics to the database.
    This helps in accumulating a history of what has been trending over time.
    """
    print(f"   [Tool] Saving Trends: {new_trends}")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    
    existing_data = []
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
        except Exception as e:
            print(f"   [Warning] Could not read existing data: {e}, starting fresh.")
            existing_data = []
            
    # Deduplicate (case-insensitive check, but allow case preservation)
    existing_set = {t.lower() for t in existing_data}
    added_count = 0
    
    for trend in new_trends:
        if trend.lower() not in existing_set:
            existing_data.append(trend)
            existing_set.add(trend.lower())
            added_count += 1
            
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)
        return f"Successfully saved {added_count} new trends. Total tracked: {len(existing_data)}."
    except Exception as e:
        return f"Error saving trends: {e}"
