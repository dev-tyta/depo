# testing model loading with joblib
import joblib
import pandas as pd

loaded_model = joblib.load('rf_classifier_model.pkl')

import numpy as np

# Example: A new data point with 30 features (as in the Wisconsin dataset)
# In a real application, this data would come from a user input, an API request, etc.
new_data = np.array([[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 0.2419, 0.07871, 
                      1.095, 0.9053, 8.589, 153.4, 0.006399, 0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 
                      25.38, 17.33, 184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]])

# Make a prediction
prediction = loaded_model.predict(new_data)

# Get the probability of the predictions
prediction_proba = loaded_model.predict_proba(new_data)

# The target names for the breast cancer dataset are typically 'malignant' and 'benign'
target_names = ['malignant', 'benign'] 

# Print the results
# The model's prediction is already the string label, so just print it.
print(f"Prediction: {prediction[0]}")
print(f"Prediction Probabilities: {prediction_proba}")