import joblib
import pandas as pd
from preprocess import preprocess_data

# Load models and data
rf_model = joblib.load('rf_model.joblib')
_, X_test, _, y_test = preprocess_data('data/UNSW_NB15_training-set.csv', 'data/UNSW_NB15_testing-set.csv')

# Take a small sample for the demo
sample_data = X_test[:5]
actual_labels = y_test[:5].values

print("\n--- Threat Detection System Live Demo ---")
predictions = rf_model.predict(sample_data)

for i in range(len(predictions)):
    status = "⚠️ ATTACK DETECTED" if predictions[i] == 1 else "✅ NORMAL TRAFFIC"
    actual = "ATTACK" if actual_labels[i] == 1 else "NORMAL"
    print(f"Packet {i+1}: {status} (Actual: {actual})")