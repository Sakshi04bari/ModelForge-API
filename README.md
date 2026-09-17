# ⚒️ ModelForge API

> A production-style REST API that serves a machine learning model — built to
> demonstrate ML engineering practice, not just model training.
>
> Repo: `modelforge-api` &nbsp;|&nbsp; Current model: Iris species classifier

[![Status](https://img.shields.io/badge/status-planning-yellow)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## 📌 Project Summary

| | |
|---|---|
| **Project** | ModelForge API — a reusable pattern for serving ML models as REST endpoints |
| **Problem type** | Multi-class classification |
| **Dataset (current model)** | [Iris](https://archive.ics.uci.edu/dataset/53/iris) — 150 samples, 4 features, 3 balanced classes |
| **Deliverable** | A `/predict` REST endpoint served over HTTP |
| **Focus** | API design, input validation, error handling, project structure — *not* model complexity |

**ModelForge** is where a raw, trained model gets shaped into a properly
engineered, deployable service. This first version wraps a small,
well-understood dataset (Iris) intentionally — the goal isn't a
state-of-the-art classifier, it's the **service** around it: a clear
contract, validated inputs, predictable responses, and a repo structure
that scales past a single notebook and past this first dataset.

---

## 🧠 Model vs. Service

A recurring theme of this project:

- **The model** is a pure function — four numbers in, one label out. It has
  no concept of HTTP, JSON, or bad input.
- **The service** is everything around it — parsing requests, validating
  data, calling the model safely, formatting a response, and returning the
  right status code when something goes wrong.

Task 1 is entirely about designing the service, before any model code exists.

---

## 📂 Dataset

**Iris flower dataset** (`iris_dataset.csv`) — the classic, widely-used
benchmark dataset for classification.

| Feature | Type | Unit |
|---|---|---|
| `sepal_length` | float | cm |
| `sepal_width` | float | cm |
| `petal_length` | float | cm |
| `petal_width` | float | cm |
| `species` (target) | categorical | `setosa` / `versicolor` / `virginica` |

- 150 rows, no missing values, 3 perfectly balanced classes.
- Chosen so data cleaning never competes for attention with API design.

### The three species

| Iris setosa | Iris versicolor | Iris virginica |
|---|---|---|
| ![Iris setosa](https://upload.wikimedia.org/wikipedia/commons/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg) | ![Iris versicolor](https://upload.wikimedia.org/wikipedia/commons/4/41/Iris_versicolor_3.jpg) | ![Iris virginica](https://upload.wikimedia.org/wikipedia/commons/9/9f/Iris_virginica.jpg) |

---

## 📜 API Contract

### `POST /predict`

**Request**

```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
```

All four fields are **required**, must be numeric, and must be positive
(measurements in centimeters).

**Response — `200 OK`**

```json
{
  "species": "setosa",
  "confidence": 0.98
}
```

**Response — `422 Unprocessable Entity`** (validation failure)

```json
{
  "error": "petal_width must be a positive number",
  "field": "petal_width"
}
```

The client sends four flower measurements; the API
returns its best-guess species and a confidence score. If the input is
missing a field, isn't numeric, or is out of a sane range, the request
fails fast with a `422` naming the exact bad field — it never silently
guesses, and it never lets bad data reach the model.

---

## 🔄 Request Lifecycle

```
Client
  │
  │ HTTP Request
  ▼
┌─────────────────────────┐
│ 1. FastAPI receives     │
│    the request          │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 2. API Key             │
│    Authentication       │
│                         │
│    ✗ Invalid/Missing    │
│      → 401 Unauthorized │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 3. Pydantic Validation │
│                         │
│ • Required fields       │
│ • Numeric values        │
│ • Range validation      │
│ • Extra-field rejection │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 4. Model Inference     │
│                         │
│ Random Forest           │
│ classifier predicts     │
│ the Iris species        │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│ 5. Response Formatting │
│                         │
│ JSON response with      │
│ prediction information  │
└────────────┬────────────┘
             ▼
       ┌─────┴─────┐
       │           │
       ▼           ▼
    Client      Monitoring
                  + Logging
```

**Key design decision:** validation is a hard gate *before* the model is
ever invoked. The model is never responsible for defending itself against
malformed input — that's the API layer's job.

---

## 🗂️ Planned Project Structure

Structured using a clean, production-ready project layout, adapted specifically for an ML API-serving project.

```
modelforge-api/

├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
│
├── app/
│   ├── main.py                 <- FastAPI application entrypoint
│   ├── config.py               <- Application configuration
│   ├── security.py             <- API key authentication
│   ├── logging_config.py       <- Logging configuration
│   ├── metrics.py              <- Prometheus custom metrics
│   │
│   ├── models/
│   │   └── schemas.py          <- Pydantic request/response schemas
│   │
│   └── routers/
│       ├── v1.py               <- Version 1 API endpoints
│       └── v2.py               <- Version 2 API endpoints
│
├── ml/
│   ├── train.py                <- Model training
│   ├── predict.py              <- Local model prediction script
│   │
│   └── saved_model/
│       ├── model.joblib        <- Trained Random Forest model
│       └── model_info.json     <- Model metadata
│
├── tests/
│   ├── test_main.py            <- API tests
│   ├── test_metrics.py         <- Metrics tests
│   └── load_test.py            <- Concurrent load testing
│
└── .github/
    └── workflows/
        └── tests.yml           <- GitHub Actions CI
```
## Preprocessing

No scaling or encoding was required for this Iris classification model because all input features are numerical and Random Forest does not require feature scaling.

If preprocessing is introduced in a future version, a scikit-learn Pipeline will be used to ensure the same preprocessing is applied during both training and prediction.
---

## 🤔 Why Iris (as the first model)?

- **Small & clean** — 150 rows, zero missing values, no data-engineering detour.
- **Well understood** — if the API misbehaves, it's the API's fault, not a confusing dataset.
- **Genuine multi-class problem** — enough complexity to exercise validation and response design, without an overbuilt model competing for attention.

ModelForge is designed so a future model swap — a different dataset, a
different classifier, even a regression problem — shouldn't require
renaming or restructuring the repo. Iris is the first tenant, not the
identity, of this project.

## ⚙️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/Sakshi04bari/ModelForge-API.git
cd ModelForge-API
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

**Windows PowerShell:**

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```
API_KEY=your-api-key
```

### 6. Run the API Locally

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```
http://localhost:8000
```

### 7. Open API Documentation

**Swagger UI:**

```
http://localhost:8000/docs
```

**ReDoc:**

```
http://localhost:8000/redoc
```

### 8. Run with Docker Compose

Build and start the application:

```bash
docker compose up --build
```

Run in detached mode:

```bash
docker compose up --build -d
```

Check running containers:

```bash
docker compose ps
```

View logs:

```bash
docker compose logs
```

Follow logs:

```bash
docker compose logs -f
```

Stop the application:

```bash
docker compose down
```

### 9. Run Tests

```bash
python -m pytest -v
```

### 10. Run Load Test

Set the API key in PowerShell:

```powershell
$env:API_KEY="your-api-key"
```

Run the load test:

```bash
python tests/load_test.py
```

### 11. Check API Health

```bash
curl http://localhost:8000/
```

Protected health endpoint:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/health
```

### 12. Test Prediction

```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key" \
  -d "{\"sepal_length\":5.1,\"sepal_width\":3.5,\"petal_length\":1.4,\"petal_width\":0.2}"
```

### 13. Check Prometheus Metrics

```
http://localhost:8000/metrics
```

Or:

```bash
curl http://localhost:8000/metrics
```

---

## 🔐 API Authentication

Protected endpoints require an API key.

The API key must be sent using the `X-API-Key` header.

**Example:**

```bash
curl -H "X-API-Key: your-api-key" \
  http://localhost:8000/api/v1/health
```

---

## 🧪 Run Tests

Run the complete test suite:

```bash
python -m pytest -v
```

**Expected result:**

```
13 passed
```