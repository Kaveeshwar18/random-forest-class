import streamlit as st
import pandas as pd
import joblib

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Cancer Prediction AI",
    page_icon="🧬",
    layout="wide"
)

# -------------------------
# Load Model
# -------------------------
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

# -------------------------
# Colorful CSS
# -------------------------
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #FF512F, #DD2476);
}

.title {
    font-size: 48px;
    font-weight: 900;
    text-align: center;
    color: white;
}

.card {
    background: rgba(255,255,255,0.15);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(15px);
    box-shadow: 0px 10px 40px rgba(0,0,0,0.3);
}

.stNumberInput label {
    color: white !important;
    font-weight: 600;
}

.stButton>button {
    background: linear-gradient(90deg, #00F260, #0575E6);
    color: white;
    font-size: 18px;
    border-radius: 15px;
    padding: 12px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
}

.result {
    padding: 30px;
    border-radius: 20px;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Title
# -------------------------
st.markdown('<p class="title">🧬 AI Cancer Prediction Dashboard</p>', unsafe_allow_html=True)
st.write("")

# -------------------------
# Dynamic Feature Inputs
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔬 Enter Medical Features")

input_data = []

for feature in features:
    value = st.number_input(f"{feature}", value=0.0, format="%.4f")
    input_data.append(value)

st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# -------------------------
# Predict Button
# -------------------------
if st.button("🚀 Predict Cancer Status"):

    input_df = pd.DataFrame([input_data], columns=features)

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.markdown(
            '<div class="result" style="background: linear-gradient(135deg, #ff416c, #ff4b2b);">⚠️ Malignant Tumor Detected</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="result" style="background: linear-gradient(135deg, #00b09b, #96c93d);">✅ Benign Tumor Detected</div>',
            unsafe_allow_html=True
        )

st.markdown("<center style='color:white;'>Developed by Kaveeshwar 🚀</center>", unsafe_allow_html=True)