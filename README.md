# 🐱🐶 Cat vs Dog Classification using SVM – SkillCraft Task 03

This project is part of my SkillCraft internship. The goal is to build a **Support Vector Machine (SVM)** model to classify images of cats and dogs using a dataset from Kaggle.

---

## 🎯 Objective

To classify images into two categories: **Cat** or **Dog** using:
- Image Preprocessing
- Feature Extraction (HOG)
- Support Vector Machine (SVM) Classifier

---

## 📁 Dataset

**Source**: [Kaggle Dogs vs Cats Dataset](https://www.kaggle.com/datasets/biaiscience/dogs-vs-cats)

- Total Images: 25,000 (12,500 Cats + 12,500 Dogs)
- Format: JPG
- Used a small subset (e.g., 1000 images) for quick training/testing

---

## 🛠️ Tools & Libraries

- Python 3
- OpenCV (`cv2`)
- Scikit-learn (`sklearn`)
- NumPy
- Matplotlib
- HOG (Histogram of Oriented Gradients)

---

## 📊 Workflow

1. **Image Preprocessing**
   - Resize images to 64x64 pixels
   - Convert to grayscale

2. **Feature Extraction**
   - Apply HOG (Histogram of Oriented Gradients)

3. **Model Building**
   - Train a Support Vector Machine (SVM) using `sklearn`

4. **Evaluation**
   - Test the model
   - Show accuracy and confusion matrix

---

## 📈 Results

- Accuracy: ~[Your accuracy]%
- Model generalizes well on unseen data

---

