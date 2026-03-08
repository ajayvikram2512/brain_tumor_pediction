import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from gradcam import make_gradcam_heatmap


@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/brain_tumor_model.keras", compile=False)

model = load_model()

classes = ["glioma","meningioma","notumor","pituitary"]

st.title("Brain Tumor Detection + Explainable AI")

uploaded_file = st.file_uploader("Upload MRI Image")

# Function to check if prediction is reliable
def is_valid_prediction(prediction):

    max_prob = np.max(prediction)
    second_prob = np.sort(prediction)[-2]

    # If probabilities are too close, model is uncertain
    if (max_prob - second_prob) < 0.15:
        return False

    return True


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI", width=300)

    img = np.array(image)

    img = np.array(image)

    img_resized = cv2.resize(img,(256,256))
    img_preprocessed = preprocess_input(img_resized)
    img_input = np.expand_dims(img_preprocessed,axis=0)

    prediction = model.predict(img_input)[0]

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    # Reject non-MRI images
    if not is_valid_prediction(prediction):

        st.error("The uploaded image does not appear to be a valid brain MRI scan.")

    else:

        st.subheader("Prediction")
        st.write("Tumor Type:", classes[class_index])
        st.write("Confidence:", round(confidence*100,2),"%")

        # Added spinner to prevent Render timeout
        with st.spinner("Generating Explainable AI visualization..."):

            heatmap = make_gradcam_heatmap(img_input, model)

            heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
            heatmap = np.uint8(255 * heatmap)

            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

            superimposed_img = heatmap * 0.4 + img

        st.subheader("Explainable AI (Grad-CAM)")
        st.image(superimposed_img.astype("uint8"))

        # Clear memory (prevents 502/503 crash on Render free tier)
        tf.keras.backend.clear_session()
