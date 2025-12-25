import json
import os
from typing import List, Type
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

DATA_FILE = "data/trends.json"

# --- Input Schemas ---

class SaveTrendsInput(BaseModel):
    new_trends: List[str] = Field(description="List of confirmed trending keywords/topics to save to the database.")

# --- Tool Classes ---

class SaveTrendsTool(BaseTool):
    name: str = "save_trends_tool"
    description: str = (
        "Save a list of confirmed trending keywords/topics to the database. "
        "This helps in accumulating a history of what has been trending over time."
    )
    args_schema: Type[BaseModel] = SaveTrendsInput

    def _run(self, new_trends: List[str]) -> str:
        print(f"   [Tool] Saving Trends: {new_trends}")
        
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        except Exception:
            # Handle case where directory might be strictly managed or current dir is weird
            pass
        
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
                # Ensure we are writing valid JSON
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
            return f"Successfully saved {added_count} new trends. Total tracked: {len(existing_data)}."
        except Exception as e:
            return f"Error saving trends: {e}"
