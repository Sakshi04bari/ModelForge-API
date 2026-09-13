from app.config import settings
def test_health(client):
    response = client.get("/api/v1/health",headers={"X-API-Key": settings.API_KEY})

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True