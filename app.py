import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import gdown

# ------------------ CONFIGURATION ------------------ #
st.set_page_config(
    page_title="Waste Classification AI",
    page_icon="♻️",
    layout="wide"
)

IMG_SIZE = 128
MODEL_PATH = "waste_classifier.keras"
FILE_ID = "1hOAC5Y5HrfKAMrqsEJJFvnAREaKYc7Zu"   
# MODEL_ACCURACY = 83.31  

# ------------------ DOWNLOAD MODEL ------------------ #
@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        url = f"https://drive.google.com/uc?id={FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_trained_model()

# ------------------ CUSTOM CSS ------------------ #
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
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------ #
st.markdown('<div class="title">♻️ Waste Classification System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI Model to Classify Waste as Organic or Recyclable</div>', unsafe_allow_html=True)

# Show Model Accuracy
# st.markdown(f"""
# <div class="metric-box">
#     <h4>📊 Model Validation Accuracy</h4>
#     <h2>{MODEL_ACCURACY}%</h2>
# </div>
# """, unsafe_allow_html=True)

# st.markdown("---")

# ------------------ FILE UPLOADER ------------------ #
uploaded_files = st.file_uploader(
    "📤 Upload one or more waste images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# ------------------ PREDICTION LOGIC ------------------ #
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

            if prediction > 0.5:
                label = "♻️ Recyclable Waste"
                confidence = round(float(prediction) * 100, 2)
            else:
                label = "🌿 Organic Waste"
                confidence = round((1 - float(prediction)) * 100, 2)

            st.success(label)
            st.progress(int(confidence))
            st.write(f"Confidence: {confidence}%")

        st.markdown("---")

else:
    st.info("Upload images to start classification.")