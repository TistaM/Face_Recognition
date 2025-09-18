import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt

def evaluate_model(model, X1_val, X2_val, y_val, threshold=0.5):
    # Ensure inputs are NumPy arrays
    X1_val = np.array(X1_val)
    X2_val = np.array(X2_val)

    # Add channel dimension if missing
    if len(X1_val.shape) == 3:
        X1_val = np.expand_dims(X1_val, axis=-1)
    if len(X2_val.shape) == 3:
        X2_val = np.expand_dims(X2_val, axis=-1)

    # Add batch dimension if needed
    if len(X1_val.shape) == 4 and X1_val.shape[0] != X2_val.shape[0]:
        X1_val = np.expand_dims(X1_val, axis=0)
        X2_val = np.expand_dims(X2_val, axis=0)

    # Confirm shapes before prediction
    print("X1 shape:", X1_val.shape)
    print("X2 shape:", X2_val.shape)

    # Prediction
    y_pred_prob = model.predict([X1_val, X2_val])
    y_pred = (y_pred_prob > threshold).astype("int32")

    # Basic Evaluation Metrics
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred)
    rec = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)

    print(f" Accuracy: {acc:.4f}")
    print(f" Precision: {prec:.4f}")
    print(f" Recall: {rec:.4f}")
    print(f" F1 Score: {f1:.4f}")

    print("\n Confusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    print("\n Classification Report:")
    print(classification_report(y_val, y_pred))

    # Optional metrics
    macro_f1 = f1_score(y_val, y_pred, average='macro')
    print(f" Macro-Average F1 Score: {macro_f1:.4f}")

    top1_accuracy = np.mean(y_pred.flatten() == y_val.flatten())
    print(f" Top-1 Accuracy: {top1_accuracy:.4f}")
