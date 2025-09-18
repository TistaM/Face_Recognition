import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import load_model
from utils.preprocess import load_and_preprocess_image
from evaluation.metrices import evaluate_model
from PIL import Image

# -------------------------
# Config
# -------------------------
MODEL_PATH = "models/siamese_model.keras"
IMAGE_SIZE = (160, 160)

# -------------------------
# Function used in model
# -------------------------
def abs_diff(tensors):
    x, y = tensors
    return tf.abs(x - y)

# -------------------------
# Display Utility
# -------------------------
def show_images(img1_path, img2_path, prediction_text):
    """Display two input images side by side with prediction label."""
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img1)
    plt.title("Image 1 (Reference)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.title("Image 2 (Distorted)")
    plt.axis("off")

    plt.suptitle(
        f"Prediction: {prediction_text}",
        fontsize=16,
        color="green" if "SAME" in prediction_text else "red"
    )
    plt.tight_layout()
    plt.show()

# -------------------------
# File-based Verification
# -------------------------
def verify_pair(img1_path, img2_path, model, threshold=0.4):
    img1 = load_and_preprocess_image(img1_path)
    img2 = load_and_preprocess_image(img2_path)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    score = model.predict([img1, img2])[0][0]
    print(f"\n Confidence Score: {score:.4f}")
    
    if score > threshold:
        result_text = " SAME person"
    else:
        result_text = " DIFFERENT persons"

    print(result_text)

    return np.array([img1.squeeze()]), np.array([img2.squeeze()]), np.array([1 if score > threshold else 0]), result_text

# ==================================================
# Live Camera Verification (Reference Path + Webcam)
# ==================================================
def preprocess_single_image(img):
    """Preprocess a raw BGR image from webcam to match training pipeline."""
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.medianBlur(img, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=-1)

def capture_from_camera():
    """Open webcam and capture one frame."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("❌ Could not access webcam.")
    
    print("\n📸 Press SPACE to capture your picture, or ESC to exit...")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow("Webcam - Press SPACE to capture", frame)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            cap.release()
            cv2.destroyAllWindows()
            return None
        elif key == 32:  # SPACE
            cap.release()
            cv2.destroyAllWindows()
            return frame

def live_verify(ref_path, model, threshold=0.4):
    """Compare reference image with webcam capture (saved as person-1.jpg)."""
    # Load and preprocess reference
    ref_img = load_and_preprocess_image(ref_path)

    # Capture distorted image via webcam
    live_img = capture_from_camera()
    if live_img is None:
        print("❌ No image captured.")
        return

    # Convert webcam frame to grayscale and save as person-1.jpg
    gray_frame = cv2.cvtColor(live_img, cv2.COLOR_BGR2GRAY)
    save_path = "person-1.jpg"
    cv2.imwrite(save_path, gray_frame)
    print(f"📂 Saved captured grayscale image as {save_path}")

    # Preprocess for model input
    live_img_processed = preprocess_single_image(live_img)

    # Expand dimensions for model input
    ref_img = np.expand_dims(ref_img, axis=0)
    live_img_processed = np.expand_dims(live_img_processed, axis=0)

    # Prediction
    pred = model.predict([ref_img, live_img_processed])[0][0]
    print(f"\n🔍 Similarity Score: {pred:.4f} (Threshold: {threshold})")

    # Decide result
    if pred >= threshold:
        result_text = " SAME person"
        print("✅ Same person detected!")
    else:
        result_text = " DIFFERENT persons"
        print("❌ Different person.")

    # Show side-by-side reference and grayscale live capture
    ref_display = Image.open(ref_path).convert("L")  # convert ref also to grayscale for consistency
    live_display = Image.open(save_path)            # open saved grayscale image

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(ref_display, cmap="gray")
    plt.title("Reference Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(live_display, cmap="gray")
    plt.title("Webcam Capture (person-1.jpg)")
    plt.axis("off")

    plt.suptitle(
        f"Prediction: {result_text}",
        fontsize=16,
        color="green" if "SAME" in result_text else "red"
    )
    plt.tight_layout()
    plt.show()


# ==================================================
# Example Usage
# ==================================================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Please train first.")
    
    model = load_model(MODEL_PATH, custom_objects={'abs_diff': abs_diff})

    print("\nChoose Mode:")
    print("1. Compare two image files")
    print("2. Compare reference image with webcam capture")
    choice = input("Enter choice (1/2): ").strip()

    if choice == "1":
        img1_path = input("Enter path of the Reference or clear image: ").strip()
        img2_path = input("Enter path of the Distorted image: ").strip()
        X1, X2, y_true, result_text = verify_pair(img1_path, img2_path, model, threshold=0.4)
        show_images(img1_path, img2_path, result_text)
        evaluate_model(model, X1, X2, y_true, threshold=0.4)

    elif choice == "2":
        img1_path = input("Enter Reference Image Path: ").strip()
        live_verify(img1_path, model, threshold=0.4)

sys.exit(0)