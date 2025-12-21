from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Trend Analyzer API is running"}
    print("✅ Health Check Passed")

from langchain_core.messages import AIMessage, HumanMessage

@patch("src.api.main.get_trend_agent")
def test_analyze_endpoint(mock_get_agent):
    # Mock the agent response to avoid real API calls during generic testing
    mock_agent_instance = mock_get_agent.return_value
    mock_agent_instance.invoke.return_value = {
        "messages": [
            HumanMessage(content="..."),
            AIMessage(content="# Mock Report\n\nTrend: Health")
        ]
    }

    response = client.post("/analyze", json={"query": "Test Query"})
    
    assert response.status_code == 200
    assert "report" in response.json()
    assert response.json()["report"] == "# Mock Report\n\nTrend: Health"
    print("✅ Analyze Endpoint Passed (Mocked)")

if __name__ == "__main__":
    test_health_check()
    test_analyze_endpoint()
