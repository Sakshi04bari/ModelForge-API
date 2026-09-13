from app.config import settings


VALID_INPUT = {
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
}


def test_missing_api_key(client):
    response = client.post(
        "/api/v1/predict",
        json=VALID_INPUT
    )

    assert response.status_code == 401


def test_invalid_api_key(client):
    response = client.post(
        "/api/v1/predict",
        json=VALID_INPUT,
        headers={
            "X-API-Key": "wrong-api-key"
        }
    )

    assert response.status_code == 401


def test_unexpected_extra_field(client):
    input_data = {
        **VALID_INPUT,
        "unexpected_field": "not_allowed"
    }

    response = client.post(
        "/api/v1/predict",
        json=input_data,
        headers={
            "X-API-Key": settings.API_KEY
        }
    )

    assert response.status_code == 422