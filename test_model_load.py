import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.joblib")
print("Loading model from:", MODEL_PATH)
model = joblib.load(MODEL_PATH)
print("Model loaded:", type(model))
row = pd.DataFrame([{
    'Rating': 4.0,
    'age': 10,
    'python_yn': 1,
    'R_yn': 0,
    'spark': 0,
    'aws': 0,
    'excel': 0,
    'job_state': 'CA'
}])
print("Input row:\n", row)
pred = model.predict(row)
print("Prediction:", pred)
