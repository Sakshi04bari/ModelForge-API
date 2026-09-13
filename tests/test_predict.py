from app.config import settings
def test_predict_valid_input(client):

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
    "/api/v1/predict",
    json=payload,
    headers={"X-API-Key": settings.API_KEY}
)

    assert response.status_code == 200

    data = response.json()

    assert "request_id" in data
    assert "prediction" in data
    assert "flower" in data
    assert "confidence" in data
    assert "model_version" in data

    assert data["flower"] in [
        "setosa",
        "versicolor",
        "virginica"
    ]

    assert 0 <= data["confidence"] <= 1


def test_predict_missing_field(client):

    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4
    }

    response = client.post(
        "/api/v1/predict",
        json=payload,
        headers={"X-API-Key": settings.API_KEY}
    )

    assert response.status_code == 422


def test_predict_invalid_value(client):

    payload = {
        "sepal_length": -5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    response = client.post(
        "/api/v1/predict",
        json=payload,
        headers={"X-API-Key": settings.API_KEY}
    )

    assert response.status_code == 422