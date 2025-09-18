
# Multiclass Face Identification in Adverse Climatic Conditions using Siamese Embedding Network and Ensemble Deep Learning

This project implements a Multiclass Face Identification in Adverse Climatic Conditions using a Siamese Embedding Network and an ensemble of models based on Siamese outputs. The goal is to determine whether two face images belong to the same person, even under distortions or image quality changes.

# Project Structure

face\_identify/
├── dataset/
│   ├── reference/                  # Clear reference images
│   └── distorted/                  # Distorted or noisy versions
├── evaluation/
│   ├── **init**.py
│   └── metrices.py                 # Evaluation metrics
├── models/
│   ├── siamese.py                  # Siamese model architecture
│   └── siamese\_model.keras         # Trained Siamese model file
├── utils/
│   ├── **init**.py
│   ├── dataset.py                  # Pair loader and label generator
│   └── preprocess.py               # Image preprocessing utilities
├── train.py                        # Train the Siamese model
├── run.py                          # Predict and visualize single pair
├── requirements.txt
└── README.md

# Getting Started

# 1. Clone the Repository

#2. **Install Dependencies**

pip install -r requirements.txt

# Dataset Format

dataset/
├── reference/
│   ├── user1.jpg
│   └── user2.jpg
├── distorted/
│   ├── user1.jpg
│   └── user2.jpg


> Images with the **same filename** in both folders are treated as **same-person pairs**.

# Model Architecture

## Siamese Neural Network

* Two shared CNN branches
* Feature comparison using absolute difference (`abs_diff`)
* Final binary classification layer

## Optional Ensemble (Simple Voting/Meta Classifier)

 Ensemble of three models of CNN, FaceNet and Resnet50

## How to Use

## Predict on a Single Image Pair

bash
python run.py


* Prompts for 2 image paths
* Displays prediction and visual result

## Train the Siamese Model

bash
python train.py


* Loads image pairs and labels from `utils/dataset.py`
* Trains and saves the model in `models/siamese_model.keras`

### 📊 Evaluate on Full Dataset

python evaluate_dataset.py


* Automatically forms positive and negative image pairs
* Computes classification metrics

---

## 📈 Evaluation Metrics

Implemented in `evaluation/metrices.py`:

* Accuracy
* Precision
* Recall
* F1 Score
* Macro-F1 Score
* Confusion Matrix

---

##  Preprocessing

Handled by `utils/preprocess.py`:

* Resize and normalize images
* Convert to grayscale (if required)
* Add batch and channel dimensions

---

## Example Output

Confidence Score: 0.8321
 SAME person


![Example output](https://via.placeholder.com/600x250?text=Image+Pair+Prediction)

---

##  Requirements

* Python 3.7+
* TensorFlow 2.x
* scikit-learn
* NumPy
* Pillow
* Matplotlib

Install all with:

bash
pip install -r requirements.txt



