import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent path for imports (if running as script)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preprocess import load_and_preprocess_image


# -------------------------
# Identity extraction
# -------------------------
def get_identity_name(filename):
    """
    Extract identity name from filename.
    Example:
        person1_img1.jpg -> person1
        person1-img2.png -> person1
    """
    name = os.path.splitext(filename)[0].lower()
    name = name.replace("-", "_")
    parts = name.split("_")
    return parts[0] if len(parts) >= 1 else name


# -------------------------
# Pair generation
# -------------------------
def generate_pairs_all_refs(reference_dir, distorted_dir, show_samples=True, n_samples=50):
    X1, X2, y = [], [], []

    # Collect reference images
    identity_to_refs = {}
    for fname in os.listdir(reference_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        identity = get_identity_name(fname)
        path = os.path.join(reference_dir, fname)
        identity_to_refs.setdefault(identity, []).append(
            load_and_preprocess_image(path)
        )

    # Collect distorted images
    identity_to_dists = {}
    for fname in os.listdir(distorted_dir):
        if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
            continue
        identity = get_identity_name(fname)
        path = os.path.join(distorted_dir, fname)
        identity_to_dists.setdefault(identity, []).append(
            load_and_preprocess_image(path)
        )

    total_pos, total_neg = 0, 0

    # Generate pairs
    for identity in tqdm(identity_to_refs, desc="Generating Pairs"):
        refs = identity_to_refs[identity]
        dists = identity_to_dists.get(identity, [])

        for anchor in refs:
            # Positive pairs
            same_images = [img for img in refs if img.tobytes() != anchor.tobytes()] + dists
            for img in same_images:
                X1.append(anchor)
                X2.append(img)
                y.append(1)
            total_pos += len(same_images)

            # Negative pairs
            other_identities = [id2 for id2 in identity_to_refs if id2 != identity]
            for neg_id in other_identities:
                neg_images = identity_to_refs[neg_id] + identity_to_dists.get(neg_id, [])
                for img in neg_images:
                    X1.append(anchor)
                    X2.append(img)
                    y.append(0)
                total_neg += len(neg_images)

    y_arr = np.array(y)
    print(f"\n=== Pair Generation (All References as Anchors) ===")
    print(f" Total Pairs: {len(y_arr)}")
    print(f" Positive Pairs: {total_pos}")
    print(f" Negative Pairs: {total_neg}")

    # Show sample pairs
    if show_samples:
        print("\n📸 Showing random sample pairs...")
        sample_idxs = random.sample(range(len(X1)), min(n_samples, len(X1)))
        for i in sample_idxs:
            show_image_pair(X1[i], X2[i], y[i])

    return np.array(X1), np.array(X2), y_arr


# -------------------------
# Visualization
# -------------------------
def show_image_pair(img1_data, img2_data, label):
    fig, axes = plt.subplots(1, 2, figsize=(5, 3))
    axes[0].imshow(img1_data.squeeze(), cmap="gray")
    axes[0].set_title("Anchor")
    axes[0].axis("off")
    axes[1].imshow(img2_data.squeeze(), cmap="gray")
    axes[1].set_title(f"Pair (Label={label})")
    axes[1].axis("off")
    plt.show()
