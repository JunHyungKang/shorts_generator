import sys
import os

# Ensure project root is in python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.agent.trend_analyzer import TrendAnalyzer

def main():
    print("🚀 Starting Trend Analysis for Seniors...")
    
    try:
        analyzer = TrendAnalyzer()
        report = analyzer.run_analysis()
        
        print("\n" + "="*50)
        print("📑 FINAL REPORT")
        print("="*50)
        print(report)
        print("="*50)
        
    except Exception as e:
        print(f"\n[ERROR] Analysis Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
