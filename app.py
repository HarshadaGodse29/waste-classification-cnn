import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

# Page configuration
st.set_page_config(page_title="Waste Classification AI", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #d9fdd3, #a8e6cf);
    }
    .title {
        text-align: center;
        font-size: 40px;
        font-weight: bold;
        color: #1b5e20;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #2e7d32;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
model = load_model("waste_classifier.keras")

IMG_SIZE = 128
MODEL_ACCURACY = 83.31  

# UI Title
st.markdown('<div class="title">♻️ Waste Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Model to Classify Waste as Organic or Recyclable</div>', unsafe_allow_html=True)

# st.write("")
# st.info(f"📊 Model Validation Accuracy: {MODEL_ACCURACY}%")

# Multiple image uploader
uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        col1, col2 = st.columns(2)

        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", width=300)

        with col2:
            image_resized = image.resize((IMG_SIZE, IMG_SIZE))
            image_array = np.array(image_resized) / 255.0
            image_array = np.expand_dims(image_array, axis=0)

            prediction = model.predict(image_array)[0][0]
            confidence = round(float(prediction) * 100, 2)

            if prediction > 0.5:
                st.success("♻️ Recyclable Waste")
                st.progress(int(confidence))
                st.write(f"Confidence: {confidence}%")
            else:
                confidence = 100 - confidence
                st.success("🌿 Organic Waste")
                st.progress(int(confidence))
                st.write(f"Confidence: {confidence}%")

        st.markdown("---")