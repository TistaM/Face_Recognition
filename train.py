import os
import numpy as np
from sklearn.model_selection import train_test_split
from models.siamese import build_siamese_network   # ✅ keep your model builder
from utils.dataset import generate_pairs_all_refs  # ✅ updated version

# -------------------------
# Config
# -------------------------
REFERENCE_DIR = r"C:\Users\tista\OneDrive\Desktop\short_references_final"
DISTORTED_DIR = r"C:\Users\tista\OneDrive\Desktop\Short_distortion_final"
MODEL_PATH = "models/siamese_model.keras"

# -------------------------
# Data preparation
# -------------------------
X1, X2, y = generate_pairs_all_refs(REFERENCE_DIR, DISTORTED_DIR, show_samples=False)

# Train/val split
X1_train, X1_val, X2_train, X2_val, y_train, y_val = train_test_split(
    X1, X2, y, test_size=0.2, random_state=42
)

# -------------------------
# Model
# -------------------------
model = build_siamese_network()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train
model.fit(
    [X1_train, X2_train],
    y_train,
    validation_data=([X1_val, X2_val], y_val),
    batch_size=16,
    epochs=5
)

# -------------------------
# Save model
# -------------------------
if not os.path.exists("models"):
    os.makedirs("models")

model.save(MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
