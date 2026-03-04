import joblib
from score import score

# 1. Load the model we just saved
model_path = "best_spam_model.joblib"
loaded_model = joblib.load(model_path)

# 2. Test Case 1: Obvious Spam
spam_text = "ur going bahamas callfreefone 08081560665 speak live operator claim either bahamas cruise of£2000 cash 18only opt txt 07786200117"
prediction, propensity = score(spam_text, loaded_model, threshold=0.5)

print("=== Test 1: Obvious Spam ===")
print(f"Text: {spam_text}")
print(f"Prediction: {prediction} (Expected: True)")
print(f"Propensity: {propensity:.4f}\n")

# 3. Test Case 2: Obvious Non-Spam (Ham)
ham_text = "Hey, are we still on for the study group tomorrow at the library?"
prediction, propensity = score(ham_text, loaded_model, threshold=0.5)

print("=== Test 2: Obvious Non-Spam ===")
print(f"Text: {ham_text}")
print(f"Prediction: {prediction} (Expected: False)")
print(f"Propensity: {propensity:.4f}")