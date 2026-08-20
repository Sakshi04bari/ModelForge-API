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
  │  POST /predict  { sepal_length, sepal_width, petal_length, petal_width }
  ▼
┌───────────────────────┐
│ 1. Request received    │  Parse JSON body
└──────────┬─────────────┘
           ▼
┌───────────────────────┐
│ 2. Validation           │  • All 4 fields present?
│                         │  • All numeric?
│                         │  • All positive / in range?
│                         │  ✗ FAIL → 422 + field-specific error (stop here)
└──────────┬─────────────┘
           ▼ (only valid input reaches this point)
┌───────────────────────┐
│ 3. Model inference      │  Pre-trained classifier predicts
│                         │  class label + probability
└──────────┬─────────────┘
           ▼
┌───────────────────────┐
│ 4. Response             │  Format as JSON → 200 OK
└──────────┬─────────────┘
           ▼
Client receives prediction
```

**Key design decision:** validation is a hard gate *before* the model is
ever invoked. The model is never responsible for defending itself against
malformed input — that's the API layer's job.

---

## 🗂️ Planned Project Structure

Structured using a clean, production-ready project layout, adapted specifically for an ML API-serving project.

```
modelforge-api/
├── README.md               <- You are here
├── requirements.txt          <- Pinned dependencies
├── pyproject.toml            <- Project metadata & tool config
│
├── data/
│   └── raw/                    <- iris_dataset.csv (immutable, original)
│
├── models/                     <- Serialized trained model (e.g. model.pkl)
│
├── notebooks/                   <- Exploration / training experiments
│   └── 1.0-eda-and-training.ipynb
│
├── src/
│   ├── config.py                <- Paths, constants, settings
│   ├── train.py                  <- Trains and serializes the model
│   ├── schemas.py                <- Request/response validation models
│   └── api/
│       ├── main.py                <- API entrypoint
│       └── routes/
│           └── predict.py           <- /predict endpoint logic
│
└── tests/
    ├── test_validation.py       <- Input validation unit tests
    └── test_api.py               <- Endpoint integration tests
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

