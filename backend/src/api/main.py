from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.agents.trend_scout.trend_deep_agent import get_trend_agent

app = FastAPI(title="Trend Analyzer Agent API")

# Configure CORS for Frontend (assuming Vite runs on port 5173)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    query: str = "Analyze the current YouTube trends for Korean seniors (60-70s) and suggest a video topic."

class AnalyzeResponse(BaseModel):
    report: str

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Trend Analyzer API is running"}

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_trends(request: AnalyzeRequest):
    print(f"📥 Received Analysis Request: {request.query}")
    try:
        agent = get_trend_agent()
        
        # Invoke agent
        result = agent.invoke({"messages": [{"role": "user", "content": request.query}]})
        last_message = result["messages"][-1]
        
        return AnalyzeResponse(report=last_message.content)
        
    except Exception as e:
        print(f"❌ Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
