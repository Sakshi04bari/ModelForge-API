from app.config import settings
def test_model_info(client):

    response = client.get(
        "/api/v1/model-info",
        headers={"X-API-Key": settings.API_KEY}
    )

    assert response.status_code == 200

    data = response.json()

    assert "model_type" in data
    assert "model_version" in data
    assert "training_date" in data
    assert "features" in data
    assert "dataset" in data

    assert data["model_type"] == "RandomForestClassifier"
    assert data["dataset"] == "Iris"