import cv2
import joblib
import numpy as np
from skimage.feature import hog
import os
import sys

model_path = 'svm_cat_dog_model_hog.pkl'

if not os.path.exists(model_path):
    print("❌ Model file not found. Please run the training script first.")
    sys.exit()

model = joblib.load(model_path)

def preprocess_image(image_path, img_size=(128, 128)):
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise ValueError("❌ Image not found or unsupported format.")
    img_gray = cv2.resize(img_gray, img_size)
    hog_features = hog(img_gray, orientations=9, pixels_per_cell=(8, 8),
                       cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return img_gray, hog_features.reshape(1, -1)

def predict_image(image_path):
    try:
        _, features = preprocess_image(image_path)
        prediction = model.predict(features)[0]
        label = "Dog 🐶" if prediction == 1 else "Cat 🐱"

        # Show result on color image
        img_color = cv2.imread(image_path)
        cv2.putText(img_color, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Prediction", img_color)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("❌ Error:", e)

# 🔍 Test it
if __name__ == "__main__":
    image_path = r"C:\Users\madha\OneDrive\Desktop\intern\task_3\test_images\download.jpg"
    predict_image(image_path)
