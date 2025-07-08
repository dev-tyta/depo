import streamlit as st
import joblib
import numpy as np
import pickle

# --- 1. Load Model and Define Feature Information ---

@st.cache_resource
def load_model():
    """
    Loads the saved model.
    Using st.cache_resource to load the model only once.
    """
    try:
        # Load the pre-trained model from a pickle file
        model = joblib.load('rf_classifier_model.pkl')
        return model
    except FileNotFoundError:
        st.error("The model file 'voting_classifier_model.pkl' was not found.")
        st.info("Please ensure you have saved your model using pickle and the file is in the same directory.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

model = load_model()


# --- Hardcoded Feature Information ---
# We define the feature names and their typical ranges (min, mean, max)
# This removes the need to load the original dataset file.
FEATURE_INFO = {
    'radius_mean': [6.98, 14.12, 28.11],
    'texture_mean': [9.71, 19.28, 39.28],
    'perimeter_mean': [43.79, 91.96, 188.5],
    'area_mean': [143.5, 654.8, 2501.0],
    'smoothness_mean': [0.05, 0.09, 0.16],
    'compactness_mean': [0.01, 0.10, 0.34],
    'concavity_mean': [0.0, 0.08, 0.42],
    'concave points_mean': [0.0, 0.04, 0.20],
    'symmetry_mean': [0.10, 0.18, 0.30],
    'fractal_dimension_mean': [0.04, 0.06, 0.09],
    'radius_se': [0.11, 0.40, 2.87],
    'texture_se': [0.36, 1.21, 4.88],
    'perimeter_se': [0.75, 2.86, 21.98],
    'area_se': [6.80, 40.33, 542.2],
    'smoothness_se': [0.001, 0.007, 0.031],
    'compactness_se': [0.002, 0.025, 0.135],
    'concavity_se': [0.0, 0.031, 0.396],
    'concave points_se': [0.0, 0.011, 0.052],
    'symmetry_se': [0.007, 0.020, 0.078],
    'fractal_dimension_se': [0.0008, 0.003, 0.029],
    'radius_worst': [7.93, 16.26, 36.04],
    'texture_worst': [12.02, 25.67, 49.54],
    'perimeter_worst': [50.41, 107.26, 251.2],
    'area_worst': [185.2, 880.5, 4254.0],
    'smoothness_worst': [0.07, 0.13, 0.22],
    'compactness_worst': [0.02, 0.25, 1.05],
    'concavity_worst': [0.0, 0.27, 1.25],
    'concave points_worst': [0.0, 0.11, 0.29],
    'symmetry_worst': [0.15, 0.29, 0.66],
    'fractal_dimension_worst': [0.05, 0.08, 0.20]
}
feature_names = list(FEATURE_INFO.keys())


# --- 2. Streamlit App Interface ---

st.set_page_config(page_title="Breast Cancer Diagnosis System", layout="wide")

# Main Title
st.title("🔬 Breast Cancer Diagnosis System Interface")
st.markdown("""
This application uses a pre-trained model to predict whether a breast tumor is **Malignant** or **Benign**.
""")

st.write("---")


# --- 3. User Input via Sliders ---

st.sidebar.header("Input Tumor Features")
st.sidebar.markdown("Use the sliders to provide the feature values.")

# Dictionary to hold the user's input
input_features = {}

# Create sliders for all features based on the hardcoded info
for feature, values in FEATURE_INFO.items():
    min_val, mean_val, max_val = values
    input_features[feature] = st.sidebar.slider(
        label=f"{feature.replace('_', ' ').title()}",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(mean_val), # Default to the mean value
        key=f"slider_{feature}"
    )

st.sidebar.write("---")


# --- 4. Prediction Logic ---

# Convert the dictionary of input features into a NumPy array
input_data = np.array([list(input_features.values())])

# Main section for displaying inputs and results
st.header("Prediction Results")
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Current Input Values")
    st.json(input_features)

# "Predict" button
if st.button("✨ Predict Diagnosis", key="predict_button"):
    try:
        # Make prediction
        prediction_label = model.predict(input_data)[0]

        # Get prediction probabilities
        prediction_proba = model.predict_proba(input_data)[0]

        with col2:
            st.subheader("Diagnosis")
            # Display the predicted label
            if str(prediction_label).upper() == 'M':
                st.error("Predicted Diagnosis: **Malignant**")
            else:
                st.success("Predicted Diagnosis: **Benign**")

            st.subheader("Prediction Confidence")
            # Get the class labels from the model itself to ensure correct order
            class_labels = list(model.classes_)
            
            # Display probabilities for each class
            for i, label in enumerate(class_labels):
                display_label = "Malignant" if str(label).upper() == 'M' else "Benign"
                st.write(f"Confidence for **{display_label}**: `{prediction_proba[i]:.2%}`")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.write("---")