# File: data_preprocessing_fer.py
# Based on data_preprocessing.py from the PDF [cite: 432-468]
# Purpose: Load and preprocess data specifically from the fer2013.csv file.

import numpy as np
import pandas as pd
# Removed cv2 import as it's not needed here
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import argparse
import os

# Original labels from PDF
EMOTION_LABELS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5: 'surprise', 6: 'neutral'}

def load_fer2013(csv_path):
    """Loads pixels and emotions from the fer2013.csv file."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Error: The file {csv_path} was not found. Please download it from Kaggle.")

    df = pd.read_csv(csv_path)
    pixels = df['pixels'].tolist()
    emotions = df['emotion'].values
    usages = df['Usage'].tolist() # Keep track of original splits if needed later, though we re-split

    X = []
    y = []
    # Process only Training and PublicTest images initially based on original structure
    # PrivateTest can be used as a final hold-out set later if desired
    for i, px in enumerate(pixels):
        # if usages[i] in ['Training', 'PublicTest']: # Optionally filter by usage if needed
            try:
                arr = np.fromstring(px, dtype=int, sep=' ')
                img = arr.reshape(48, 48)
                X.append(img)
                y.append(emotions[i])
            except ValueError as e:
                print(f"Warning: Skipping row {i} due to parsing error: {e}. Pixel string: '{px[:50]}...'")

    X = np.array(X, dtype='uint8')
    y = np.array(y)

    print(f"Loaded {len(X)} images from {csv_path}")
    return X, y

def preprocess_images(X):
    """Normalizes images and adds channel dimension."""
    X = X.astype('float32') / 255.0
    X = np.expand_dims(X, -1) # Add channel dimension for grayscale
    return X

# Keep this main block if you want to run preprocessing separately
# and save to NPZ, though the combined script handles it internally.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess fer2013.csv and save splits to NPZ")
    parser.add_argument('--csv', required=True, help='Path to fer2013.csv file')
    parser.add_argument('--out', default='data/fer_processed.npz', help='Output path for the .npz file')
    parser.add_argument('--val_size', type=float, default=0.15, help='Proportion of data to use for validation set')

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    X, y = load_fer2013(args.csv)
    X = preprocess_images(X)
    y_cat = to_categorical(y, num_classes=len(EMOTION_LABELS_FER))

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat,
        test_size=args.val_size,
        random_state=42,
        stratify=y  # Ensure class distribution is similar in train/val
    )

    np.savez_compressed(args.out, X_train=X_train, X_val=X_val,
                        y_train=y_train, y_val=y_val)

    print(f"Saved processed FER data -> {args.out}")
    print(f"Shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")
