import joblib
from flask import Flask, request, jsonify
from score import score

app = Flask(__name__)

# Load the model once when the app starts
MODEL_PATH = "best_spam_model.joblib"
model = joblib.load(MODEL_PATH)

@app.route('/score', methods=['POST'])
def score_endpoint():
    """
    Flask endpoint that receives a text message via POST request
    and returns the spam prediction and propensity in JSON format.
    """
    # Parse the incoming JSON data
    data = request.get_json(force=True)
    
    # Extract the text and an optional threshold (defaults to 0.5)
    text = data.get('text', '')
    threshold = data.get('threshold', 0.5)
    
    # Run the text through our score function
    prediction, propensity = score(text, model, threshold)
    
    # Return the results as a JSON response
    return jsonify({
        'prediction': prediction,
        'propensity': propensity
    })

if __name__ == '__main__':  # pragma: no cover
    # Run the Flask development server on localhost, port 5000
    app.run(host='0.0.0.0', port=5000)