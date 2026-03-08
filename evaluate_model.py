import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

test_dir = "dataset/Testing"

IMG_SIZE = 256
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

model = tf.keras.models.load_model("model/brain_tumor_model.keras")

pred = model.predict(test_data)
pred_classes = np.argmax(pred, axis=1)

print(classification_report(test_data.classes, pred_classes))
print(confusion_matrix(test_data.classes, pred_classes))