def test_v2_predict(client):

    response = client.post(
        "/api/v2/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "request_id" in data
    assert "prediction" in data
    assert "flower" in data
    assert "probabilities" in data
    assert "model_version" in data

    assert "confidence" not in data

    assert "setosa" in data["probabilities"]
    assert "versicolor" in data["probabilities"]
    assert "virginica" in data["probabilities"]
    
def test_v1_and_v2_have_different_response_shapes(client):

    input_data = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2
    }

    v1_response = client.post(
        "/api/v1/predict",
        json=input_data
    )

    v2_response = client.post(
        "/api/v2/predict",
        json=input_data
    )

    assert v1_response.status_code == 200
    assert v2_response.status_code == 200

    v1_data = v1_response.json()
    v2_data = v2_response.json()

    # V1 uses confidence
    assert "confidence" in v1_data
    assert "probabilities" not in v1_data

    # V2 uses probabilities
    assert "probabilities" in v2_data
    assert "confidence" not in v2_data    
    
        # Both predictions are valid
    assert v1_data["prediction"] in [0, 1, 2]
    assert v2_data["prediction"] in [0, 1, 2]

    # V2 version
    assert v2_data["model_version"] == "2.0"