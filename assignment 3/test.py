import joblib
import pytest
import subprocess
import time
import requests
from score import score
import sys
import app  # We import app here so pytest-cov can see it!

# Load the model once for all tests
MODEL_PATH = "best_spam_model.joblib"
model = joblib.load(MODEL_PATH)

def test_score():
    """Unit test for the score function covering all assignment requirements."""
    spam_text = "ur going bahamas callfreefone 08081560665 speak live operator claim either bahamas cruise of£2000 cash 18only opt txt 07786200117"
    ham_text = "Hey, are we still on for the study group tomorrow at the library?"
    
    # 1. Smoke test
    try:
        prediction, propensity = score(spam_text, model, 0.5)
    except Exception as e:
        pytest.fail(f"Function crashed during execution: {e}")
        
    # 2. Format test
    assert isinstance(prediction, bool)
    assert isinstance(propensity, float)
    
    # 3. Sanity check: boolean
    assert prediction in [True, False]
    
    # 4. Sanity check: bounds
    assert 0.0 <= propensity <= 1.0
    
    # 5. Edge case: threshold 0
    pred_t0, _ = score(ham_text, model, 0.0)
    assert pred_t0 is True
    
    # 6. Edge case: threshold 1
    pred_t1, _ = score(spam_text, model, 1.0)
    assert pred_t1 is False
    
    # 7. Typical input: spam
    pred_spam, _ = score(spam_text, model, 0.5)
    assert pred_spam is True
    
    # 8. Typical input: ham
    pred_ham, _ = score(ham_text, model, 0.5)
    assert pred_ham is False

def test_flask_client():
    """
    Test using Flask's built-in test client to ensure pytest-cov 
    can track the coverage inside app.py.
    """
    client = app.app.test_client()
    sample_data = {
        "text": "ur going bahamas callfreefone 08081560665 speak live operator claim either bahamas cruise of£2000 cash 18only opt txt 07786200117",
        "threshold": 0.5
    }
    response = client.post('/score', json=sample_data)
    assert response.status_code == 200
    assert response.json['prediction'] is True

def test_flask():
    """
    Integration test that launches the Flask app via command line, 
    as explicitly required by the assignment.
    """
    # 1. Launch the Flask app using the EXACT same python executable running this test
    process = subprocess.Popen([sys.executable, "app.py"])
    
    # Give it a bit more time to start up securely
    time.sleep(5) 
    
    try:
        # 2. Test the response from the localhost endpoint
        url = 'http://127.0.0.1:5000/score'
        sample_data = {
            "text": "ur going bahamas callfreefone 08081560665 speak live operator claim either bahamas cruise of£2000 cash 18only opt txt 07786200117",
            "threshold": 0.5
        }
        
        response = requests.post(url, json=sample_data)
        assert response.status_code == 200
        
        response_json = response.json()
        assert 'prediction' in response_json
        assert 'propensity' in response_json
        assert response_json['prediction'] is True
        
    finally:
        # 3. Close the flask app using command line equivalent
        process.terminate()
        process.wait()