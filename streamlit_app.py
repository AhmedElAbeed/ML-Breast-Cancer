import pickle
import numpy as np
import streamlit as st
from pathlib import Path
from streamlit.components.v1 import html

# Load the model
MODEL_PATH = 'models/breast_cancer.pkl'
model = pickle.load(open(MODEL_PATH, 'rb'))

# Prediction function
def predict(values):
    values = np.asarray(values).reshape(1, -1)
    return model.predict(values)[0]

# Function to load HTML content as string
def load_html(file_path):
    try:
        # Read the HTML file and return its content
        return Path(file_path).read_text(encoding="utf-8")
    except Exception as e:
        st.error(f"Error loading HTML file: {e}")
        return None

# Streamlit App
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Breast Cancer Prediction"])

if page == "Home":
    st.title("Home")
    home_html = load_html("templates/home.html")
    if home_html:
        # Render the Home page HTML
        html(home_html, height=800, scrolling=True)
    else:
        st.error("Failed to load Home page HTML")

elif page == "Breast Cancer Prediction":
    st.title("Breast Cancer Prediction")

    # Load and display the Breast Cancer Prediction page HTML
    breast_cancer_html = load_html("templates/breast_cancer.html")
    if breast_cancer_html:
        # Render the Breast Cancer Prediction page HTML
        html(breast_cancer_html, height=800, scrolling=True)
    else:
        st.error("Failed to load Breast Cancer Prediction page HTML")

    # Prediction form and logic
    st.write("Fill out the form and click Predict.")
    user_data = [st.number_input(f"Feature {i + 1}:", value=0.0, step=0.01) for i in range(22)]

    if st.button("Predict"):
        try:
            prediction = predict(user_data)
            result = "Positive for Breast Cancer" if prediction == 1 else "Negative for Breast Cancer"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error(f"Error: {str(e)}")
