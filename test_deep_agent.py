import sys
import os

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.trend_deep_agent import get_trend_agent

def main():
    print("🚀 Starting Deep Agent for Trend Analysis...")
    print("   (Using ModelFallbackMiddleware for robust execution)")
    
    try:
        agent = get_trend_agent()
        
        # User query to kick off the autonomous workflow
        query = "Analyze the current YouTube trends for Korean seniors (60-70s) and suggest a video topic."
        print(f"\nUser Query: {query}\n")
        
        # Invoke the agent
        result = agent.invoke({"messages": [{"role": "user", "content": query}]})
        
        # The result returns the final state. The last message is usually the answer.
        last_message = result["messages"][-1]
        
        print("\n" + "="*50)
        print("🤖 AGENT REPORT")
        print("="*50)
        print(last_message.content)
        print("="*50)
        
    except Exception as e:
        print(f"\n[ERROR] Agent Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
