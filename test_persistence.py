import os
import json
import shutil
from src.tools.storage_tools import save_trends_tool, DATA_FILE

def test_save_trends():
    print("🧪 Testing Persistence Logic...")
    
    # 1. Clean up old data
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)
        
    # 2. First Save
    keywords_1 = ["Health", "Trot", "Finance"]
    result_1 = save_trends_tool.invoke({"new_trends": keywords_1})
    print(f"   Save 1 Result: {result_1}")
    
    assert os.path.exists(DATA_FILE)
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    assert len(data) == 3
    assert "Health" in data
    
    # 3. Second Save (Duplicate Check)
    keywords_2 = ["health", "Travel", "TROT"] # duplicates with case variation
    result_2 = save_trends_tool.invoke({"new_trends": keywords_2})
    print(f"   Save 2 Result: {result_2}")
    
    with open(DATA_FILE, "r") as f:
        data = json.load(f)
    
    # "Health" and "Trot" should not be added again (case insensitive logic in tool)
    # But tool implementation: "existing_set = {t.lower() for t in existing_data}"
    # So "health" (lower) should be found in existing set.
    # New trends: "Travel". Total should be 4.
    
    assert len(data) == 4 
    assert "Travel" in data
    
    print("✅ Persistence Logic Verified!")
    
    # Clean up
    if os.path.exists(DATA_FILE):
        os.remove(DATA_FILE)

if __name__ == "__main__":
    test_save_trends()
