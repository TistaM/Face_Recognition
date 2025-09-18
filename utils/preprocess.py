import os
import random
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt

IMAGE_SIZE = (160, 160)

def load_and_preprocess_image(path):
    """
    Load and preprocess a single image from a full file path.
    Args:
        path (str): full file path to the image
    Returns:
        np.ndarray: preprocessed image (H, W, 1)
    """
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"❌ Image not found: {path}")

    # Resize
    img = cv2.resize(img, IMAGE_SIZE)

    # Filtering
    img = cv2.medianBlur(img, 3)
    img = cv2.GaussianBlur(img, (3, 3), 0)

    # Grayscale + Normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("float32") / 255.0

    return np.expand_dims(img, axis=-1)
