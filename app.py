import streamlit as st
import pickle
import numpy as np
import pandas as pd

import os

@st.cache_resource
def load_model():
    # Get the directory where app.py is located
    base_path = os.path.dirname(os.path.abspath(__file__))
    
    model_path  = os.path.join(base_path, "model.pkl")
    scaler_path = os.path.join(base_path, "scaler.pkl")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model()

st.title("🔬 Breast Cancer Prediction")
st.markdown("Upload a CSV file with tumor measurements to get a prediction.")

# Download sample CSV button
sample_data = pd.DataFrame([{
    "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8,
    "area_mean": 1001.0, "smoothness_mean": 0.1184, "compactness_mean": 0.2776,
    "concavity_mean": 0.3001, "concave points_mean": 0.1471,
    "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095, "texture_se": 0.9053, "perimeter_se": 8.589,
    "area_se": 153.4, "smoothness_se": 0.006399, "compactness_se": 0.04904,
    "concavity_se": 0.05373, "concave points_se": 0.01587,
    "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.6,
    "area_worst": 2019.0, "smoothness_worst": 0.1622, "compactness_worst": 0.6656,
    "concavity_worst": 0.7119, "concave points_worst": 0.2654,
    "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.1189
}])

st.download_button(
    label="📥 Download Sample CSV",
    data=sample_data.to_csv(index=False),
    file_name="sample_input.csv",
    mime="text/csv"
)

st.divider()

# File uploader
uploaded_file = st.file_uploader("📂 Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Uploaded Data Preview", df.head())

    if df.shape[1] != 30:
        st.error(f"❌ Expected 30 features, got {df.shape[1]}. Please check your CSV.")
    else:
        features_scaled = scaler.transform(df.values)
        predictions     = model.predict(features_scaled)
        probabilities   = model.predict_proba(features_scaled)

        df["Prediction"]          = ["✅ Benign" if p == 1 else "⚠️ Malignant" for p in predictions]
        df["Benign Confidence"]    = [f"{p[1]*100:.2f}%" for p in probabilities]
        df["Malignant Confidence"] = [f"{p[0]*100:.2f}%" for p in probabilities]

        st.divider()
        st.write("### 🧾 Prediction Results")
        st.dataframe(df[["Prediction", "Benign Confidence", "Malignant Confidence"]])

        # Summary
        benign_count    = sum(predictions == 1)
        malignant_count = sum(predictions == 0)
        col1, col2 = st.columns(2)
        col1.metric("✅ Benign Cases",    benign_count)
        col2.metric("⚠️ Malignant Cases", malignant_count)

        st.warning("⚕️ For educational purposes only. Always consult a medical professional.")
