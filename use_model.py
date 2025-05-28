import cv2
import numpy as np
from tensorflow.keras.models import load_model

def detect_deepfake(image_path, model_path="best_model_xception.h5"):
    model = load_model(model_path)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224)) / 255.0
    img = img.reshape(1, 224, 224, 3)
    pred = model.predict(img)[0][0]
    return "Fake" if pred > 0.5 else "Real"

# usage
if __name__ == "__main__":
    result = detect_deepfake("sample_image.jpg")
    print("Result:", result)