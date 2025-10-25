# data_preprocessing.py
import numpy as np
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import argparse

EMOTION_LABELS = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'sad', 5:
'surprise', 6: 'neutral'}

def load_fer2013(csv_path):
    df = pd.read_csv(csv_path)
    pixels = df['pixels'].tolist()
    emotions = df['emotion'].values
    X = np.zeros((len(pixels), 48, 48), dtype='uint8')
    for i, px in enumerate(pixels):
        arr = np.fromstring(px, dtype=int, sep=' ')
        X[i] = arr.reshape(48, 48)
    y = emotions
    return X, y

def preprocess_images(X):
    X = X.astype('float32') / 255.0
    X = np.expand_dims(X, -1)
    return X

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', required=True)
    parser.add_argument('--out', default='data/processed.npz')
    args = parser.parse_args()
    X, y = load_fer2013(args.csv)
    X = preprocess_images(X)
    y_cat = to_categorical(y, num_classes=7)
    X_train, X_test, y_train, y_test = train_test_split(X, y_cat,
    test_size=0.15, random_state=42, stratify=y)
    np.savez_compressed(args.out, X_train=X_train, X_test=X_test,
    y_train=y_train, y_test=y_test)
    print('Saved processed data ->', args.out)