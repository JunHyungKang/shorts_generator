import sys
import os

# Ensure project root is in python path to import src
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.openrouter import get_chat_model

def main():
    print("Testing OpenRouter Connection...")
    
    try:
        # 1. Initialize Model
        print("Initializing model with Auto-Fallback (Free Models)...")
        from src.utils.openrouter import get_model_candidates
        candidates = get_model_candidates()
        print(f"Candidates found: {len(candidates)}")
        if not candidates:
             print("[Warning] No candidates found, check OpenRouter API status.")
        
        llm = get_chat_model(use_free_fallback=True)
        
        # 2. Send Prompt
        prompt = "Hello! Please introduce yourself briefly in Korean."
        print(f"\nSending prompt: '{prompt}'\n")
        
        response = llm.invoke(prompt)
        
        # 3. Print Response
        print("-" * 50)
        print("Response from OpenRouter:")
        print(response.content)
        print("-" * 50)
        print("\nTest Successfully Completed!")
        
    except Exception as e:
        print(f"\n[ERROR] Test Failed: {e}")
        print("Please checks your .env file and API key.")

if __name__ == "__main__":
    main()
