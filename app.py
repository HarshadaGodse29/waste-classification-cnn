import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown

# ==============================
# PAGE CONFIGURATION
# ==============================
st.set_page_config(
    page_title="Waste Classification AI",
    layout="wide"
)

# ==============================
# CUSTOM CSS
# ==============================
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #d9fdd3, #a8e6cf);
    }
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #1b5e20;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #2e7d32;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================
# MODEL DOWNLOAD CONFIG
# ==============================
MODEL_PATH = "waste_classifier.h5"
FILE_ID = "1Z3ZosG9nkZ6PWNnC_U405dxI3VjDVMy3"

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_trained_model()

IMG_SIZE = 128
# MODEL_ACCURACY = 83.31  

# ==============================
# UI HEADER
# ==============================
st.markdown('<div class="title">♻️ Waste Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Model to Classify Waste as Organic or Recyclable</div>', unsafe_allow_html=True)

# st.info(f"📊 Model Validation Accuracy: {MODEL_ACCURACY}%")

# ==============================
# MULTIPLE IMAGE UPLOAD
# ==============================
uploaded_files = st.file_uploader(
    "Upload one or more images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ==============================
# PREDICTION LOGIC
# ==============================
if uploaded_files:
    for uploaded_file in uploaded_files:
        col1, col2 = st.columns(2)

        with col1:
            image = Image.open(uploaded_file).convert("RGB")
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