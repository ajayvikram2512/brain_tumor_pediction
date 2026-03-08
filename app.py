import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from gradcam import make_gradcam_heatmap


# -----------------------------
# Load Model (cached)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/brain_tumor_model.keras", compile=False)

model = load_model()

classes = ["glioma","meningioma","notumor","pituitary"]


# -----------------------------
# UI
# -----------------------------
st.title("Brain Tumor Detection + Explainable AI")

st.write(
"""
Upload a **Brain MRI image** to detect tumor type using a Deep Learning model.
The system also highlights the **tumor region using Grad-CAM Explainable AI**.
"""
)

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])


# -----------------------------
# Function to validate prediction
# -----------------------------
def is_valid_prediction(prediction):

    max_prob = np.max(prediction)
    second_prob = np.sort(prediction)[-2]

    # If probabilities too close -> uncertain prediction
    if (max_prob - second_prob) < 0.15:
        return False

    return True


# -----------------------------
# Prediction Pipeline
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded MRI", width=300)

    img = np.array(image)

    # Resize image
    img_resized = cv2.resize(img,(256,256))

    # Preprocess
    img_preprocessed = preprocess_input(img_resized)

    img_input = np.expand_dims(img_preprocessed,axis=0)

    with st.spinner("Analyzing MRI Image..."):

        prediction = model.predict(img_input)[0]

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)


    # -----------------------------
    # Reject Non-MRI Images
    # -----------------------------
    if not is_valid_prediction(prediction):

        st.error("The uploaded image does not appear to be a valid brain MRI scan.")

    else:

        st.subheader("Prediction")

        st.write("Tumor Type:", classes[class_index])
        st.write("Confidence:", round(confidence*100,2),"%")

        # -----------------------------
        # Grad-CAM Explainable AI
        # -----------------------------
        heatmap = make_gradcam_heatmap(img_input, model)

        heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
        heatmap = np.uint8(255 * heatmap)

        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        superimposed_img = heatmap * 0.4 + img

        st.subheader("Explainable AI (Grad-CAM)")

        st.image(superimposed_img.astype("uint8"))
