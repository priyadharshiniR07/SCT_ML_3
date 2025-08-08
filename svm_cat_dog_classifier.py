import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def extract_hog_features(image_path, img_size=(128, 128)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, img_size)
    features = hog(img, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return features

def load_dataset(cat_folder, dog_folder, limit_per_class=1000):
    X, y = [], []

    print("📥 Loading Cat images...")
    for i, file in enumerate(os.listdir(cat_folder)):
        if i >= limit_per_class:
            break
        path = os.path.join(cat_folder, file)
        features = extract_hog_features(path)
        if features is not None:
            X.append(features)
            y.append(0)

    print("📥 Loading Dog images...")
    for i, file in enumerate(os.listdir(dog_folder)):
        if i >= limit_per_class:
            break
        path = os.path.join(dog_folder, file)
        features = extract_hog_features(path)
        if features is not None:
            X.append(features)
            y.append(1)

    return np.array(X), np.array(y)

# 🐱🐶 Dataset paths
cat_dir = r'C:\Users\madha\OneDrive\Desktop\intern\task_3\PetImages\Cat'
dog_dir = r'C:\Users\madha\OneDrive\Desktop\intern\task_3\PetImages\Dog'

# 📊 Load and split data
X, y = load_dataset(cat_dir, dog_dir, limit_per_class=1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train SVM
print("🔧 Training SVM model...")
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

# 📈 Evaluate
y_pred = model.predict(X_test)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 💾 Save model
joblib.dump(model, 'svm_cat_dog_model_hog.pkl')
print("✅ Model saved as 'svm_cat_dog_model_hog.pkl'")
