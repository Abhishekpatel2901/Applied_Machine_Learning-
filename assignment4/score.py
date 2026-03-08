import typing
import sklearn.base

def score(text: str, model: sklearn.base.BaseEstimator, threshold: float) -> typing.Tuple[bool, float]:
    """
    Scores a text using a trained model.
    Returns a tuple of (prediction, propensity).
    """
    # predict_proba returns an array of shape (n_samples, n_classes)
    # [0] gets the probabilities for our single input text
    probabilities = model.predict_proba([text])[0]
    
    # We assume index 1 is the probability of the positive class (spam)
    propensity = float(probabilities[1])
    
    # Prediction is True (1) if propensity meets or exceeds the threshold
    prediction = bool(propensity >= threshold)
    
    return prediction, propensity