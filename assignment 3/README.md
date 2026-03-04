# Assignment 3: Testing & Model Serving

## Overview
This repository contains the deployment and testing infrastructure for an SMS Spam Classification model. The project takes a pre-trained machine learning pipeline (TF-IDF Vectorizer + Random Forest Classifier) and serves it via a Flask REST API. It also includes a robust, 100% coverage test suite using `pytest` to ensure model stability and API reliability.

## Files Included
* **`score.py`**: Contains the core `score(text, model, threshold)` function. It processes input text and returns a boolean prediction and a propensity score (probability).
* **`app.py`**: A Flask web application that loads the trained model and exposes a `/score` POST endpoint for real-time predictions.
* **`test.py`**: A comprehensive test suite containing:
  * Unit tests verifying output formats, bounds, and edge cases for the `score` function.
  * An integration test (`test_flask`) that dynamically spins up the Flask server, sends an HTTP request, verifies the JSON response, and safely terminates the process.
* **`best_spam_model.joblib`**: The serialized, fine-tuned Random Forest model.
* **`train.ipynb`**: The Jupyter Notebook documenting the data loading, model training, grid search hyperparameter tuning, and evaluation process.
* **`coverage.txt`**: The output report from `pytest-cov`, proving 100% code execution coverage across `score.py` and `app.py`.

## Prerequisites
To run this project, ensure you have the following Python packages installed:

```bash
pip install pandas scikit-learn flask pytest pytest-cov requests joblib
```

## How to Run the Tests
This project uses `pytest` for all unit and integration testing.

To run the standard test suite:

```bash
pytest test.py -v
```

To generate the coverage report:

```bash
pytest test.py --cov=score --cov=app > coverage.txt
```

## How to Run the Flask API
1. Start the server from your terminal:

```bash
python app.py
```

2. The server will run on `http://127.0.0.1:5000/`. You can test the endpoint by sending a POST request to `/score` with a JSON payload:

```json
{
    "text": "ur going bahamas callfreefone 08081560665 speak live operator...",
    "threshold": 0.5
}
```

3. The API will return a JSON response containing the model's prediction and propensity score.