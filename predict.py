import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input

model = tf.keras.models.load_model("model/brain_tumor_model.keras")

classes = ["glioma","meningioma","notumor","pituitary"]

CONFIDENCE_THRESHOLD = 0.60

def predict_image(image):

    img = cv2.resize(image,(256,256))
    img = preprocess_input(img)
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)[0]

    class_index = np.argmax(prediction)
    confidence = np.max(prediction)

    if confidence < CONFIDENCE_THRESHOLD:
        return "Invalid Image (Not Brain MRI)", confidence

    return classes[class_index], confidence