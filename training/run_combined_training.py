# File: run_combined_training_iitm_filenames.py
# Version 2: Updated to scan IITM subfolders

import argparse
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import datetime
from tqdm import tqdm # For progress bar

# Import model definition and FER processing function
from models_updated import build_emotion_cnn
from data_preprocessing_fer import load_fer2013, preprocess_images as preprocess_fer_images, EMOTION_LABELS_FER

# --- Constants ---
IMG_HEIGHT = 48 # Target size for the CNN model
IMG_WIDTH = 48
BATCH_SIZE = 64
NUM_CLASSES_FER = 7
NUM_CLASSES_IITM = 6 # NE, SA, SM, SO, SU, YN

# Map IITM filename codes to labels and numerical indices
IITM_EMOTION_MAP = {
    'NE': 0, # Neutral
    'SA': 1, # Sad
    'SM': 2, # Smile (Happy)
    'SO': 3, # Surprise Open
    'SU': 4, # Surprise
    'YN': 5  # Yawning
}
EMOTION_LABELS_IITM = {v: k for k, v in IITM_EMOTION_MAP.items()}

# --- Stage 1: FER Training Function (Unchanged from previous version) ---
def train_fer_model(args):
    """Trains the initial model on the FER2013 dataset."""
    print("\n" + "="*30)
    print(" Stage 1: Training on FER2013")
    print("="*30 + "\n")
    try:
        X, y = load_fer2013(args.fer_csv)
    except FileNotFoundError as e:
        print(e)
        return None
    X = preprocess_fer_images(X)
    y_cat = to_categorical(y, num_classes=NUM_CLASSES_FER)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.15, random_state=42, stratify=y
    )
    print(f"FER2013 Data shapes: X_train={X_train.shape}, y_train={y_train.shape}, X_val={X_val.shape}, y_val={y_val.shape}")

    print("Building FER2013 model...")
    model = build_emotion_cnn(input_shape=(IMG_HEIGHT, IMG_WIDTH, 1), num_classes=NUM_CLASSES_FER)
    model.compile(optimizer=Adam(learning_rate=args.lr_fer), loss='categorical_crossentropy', metrics=['accuracy'])

    datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
    datagen.fit(X_train)

    log_dir_fer = os.path.join("logs", "fit_fer", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks_fer = [
        ModelCheckpoint(args.fer_output_model, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
        TensorBoard(log_dir=log_dir_fer, histogram_freq=1)
    ]

    print(f"\nStarting FER2013 training for {args.epochs_fer} epochs...")
    print(f"TensorBoard logs: tensorboard --logdir {log_dir_fer}")
    model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(X_val, y_val),
        epochs=args.epochs_fer,
        callbacks=callbacks_fer,
        steps_per_epoch=max(1, len(X_train) // BATCH_SIZE),
        validation_steps=max(1, len(X_val) // BATCH_SIZE)
    )
    print(f"\nFinished training FER2013. Best model saved to {args.fer_output_model}")
    return args.fer_output_model

# --- UPDATED: IITM Data Loading Function ---
def load_iitm_data_from_filenames(iitm_dir):
    """Scans directory and its subdirectories, parses filenames, loads images, and returns X, y arrays."""
    images = []
    labels = []
    print(f"Scanning IITM directory tree: {iitm_dir}")
    if not os.path.isdir(iitm_dir):
         raise FileNotFoundError(f"IITM directory '{iitm_dir}' not found.")

    image_files_found = []
    # *** Use os.walk to scan recursively ***
    for root, _, files in os.walk(iitm_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_files_found.append(os.path.join(root, filename)) # Store full path

    if not image_files_found:
        raise ValueError(f"No image files found in '{iitm_dir}' or its subdirectories. Check the path and dataset structure.")

    print(f"Found {len(image_files_found)} potential image files. Parsing filenames and loading images...")

    for img_path in tqdm(image_files_found):
        filename = os.path.basename(img_path) # Get just the filename for parsing
        # Expected format: SUB<id><EmotionCode><OrientationCode>.jpg/png/...
        # Extract emotion code (2 letters after SUB<id>)
        try:
            # Find the part after "SUB" and before the orientation code
            parts = filename.split('.')[0].replace('SUB', '') # Remove SUB and extension
            # Assume ID is numeric, find where letters start
            emotion_start_index = -1
            for i, char in enumerate(parts):
                if char.isalpha():
                    emotion_start_index = i
                    break

            if emotion_start_index != -1 and len(parts) >= emotion_start_index + 3: # Need emotion code (2 chars) + orientation (1 char)
                 emotion_code = parts[emotion_start_index : emotion_start_index + 2]

                 if emotion_code in IITM_EMOTION_MAP:
                    label = IITM_EMOTION_MAP[emotion_code]

                    # Load image, convert to grayscale, resize (using the full img_path)
                    img = load_img(img_path, color_mode='grayscale', target_size=(IMG_HEIGHT, IMG_WIDTH))
                    img_array = img_to_array(img)

                    images.append(img_array)
                    labels.append(label)
                 else:
                    print(f"Warning: Unknown emotion code '{emotion_code}' in filename '{filename}'. Skipping.")
            else:
                 print(f"Warning: Could not parse emotion code from filename '{filename}'. Skipping.")

        except Exception as e:
            print(f"Error processing file {filename} at {img_path}: {e}. Skipping.")

    if not images:
         raise ValueError("No valid images could be loaded and parsed from the IITM directory tree.")

    # Convert to NumPy arrays and normalize images
    X_iitm = np.array(images, dtype='float32') / 255.0
    y_iitm = np.array(labels)

    print(f"Successfully loaded {len(X_iitm)} images for IITM dataset.")
    return X_iitm, y_iitm


# --- Stage 2: IITM Fine-tuning Function (Unchanged logic, uses output from updated load function) ---
def finetune_iitm_model(args, fer_model_path):
    """Loads the FER-trained model and fine-tunes it on the IITM dataset."""
    print("\n" + "="*30)
    print(" Stage 2: Fine-tuning on IITM")
    print("="*30 + "\n")

    # 1. Load IITM data using the updated function
    try:
        X_iitm, y_iitm = load_iitm_data_from_filenames(args.iitm_dir)
        y_iitm_cat = to_categorical(y_iitm, num_classes=NUM_CLASSES_IITM)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading IITM data: {e}")
        return None

    # Split IITM data
    X_train_iitm, X_val_iitm, y_train_iitm, y_val_iitm = train_test_split(
        X_iitm, y_iitm_cat, test_size=0.20, random_state=42, stratify=y_iitm
    )
    print(f"IITM Data shapes: X_train={X_train_iitm.shape}, y_train={y_train_iitm.shape}, X_val={X_val_iitm.shape}, y_val={y_val_iitm.shape}")

    # 2. Load the pre-trained FER model
    print(f"Loading base model from {fer_model_path}...")
    try:
        base_model = load_model(fer_model_path)
    except Exception as e:
        print(f"Error loading base model: {e}")
        return None

    # 3. Freeze base layers and add new head
    print("Freezing base model layers and adding new head...")
    for layer in base_model.layers:
        layer.trainable = False
    base_output = base_model.layers[-2].output
    new_output = Dense(NUM_CLASSES_IITM, activation='softmax', name='iitm_output')(base_output)
    finetune_model = Model(inputs=base_model.input, outputs=new_output)

    # 4. Compile the fine-tuning model
    print("Compiling fine-tuning model...")
    finetune_model.compile(optimizer=Adam(learning_rate=args.lr_iitm), loss='categorical_crossentropy', metrics=['accuracy'])

    # 5. Set up Data Augmentation
    datagen_iitm = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.15, height_shift_range=0.15,
        shear_range=0.1, zoom_range=0.1, horizontal_flip=True, fill_mode='nearest'
    )
    datagen_iitm.fit(X_train_iitm)

    # 6. Set up Callbacks
    log_dir_iitm = os.path.join("logs", "fit_iitm", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    callbacks_iitm = [
        ModelCheckpoint(args.iitm_output_model, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1),
        EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        TensorBoard(log_dir=log_dir_iitm, histogram_freq=1)
    ]

    # 7. Fine-tune the model
    print(f"\nStarting IITM fine-tuning for {args.epochs_iitm} epochs...")
    print(f"TensorBoard logs: tensorboard --logdir {log_dir_iitm}")
    history_iitm = finetune_model.fit(
        datagen_iitm.flow(X_train_iitm, y_train_iitm, batch_size=BATCH_SIZE),
        validation_data=(X_val_iitm, y_val_iitm),
        epochs=args.epochs_iitm,
        callbacks=callbacks_iitm,
        steps_per_epoch=max(1, len(X_train_iitm) // BATCH_SIZE),
        validation_steps=max(1, len(X_val_iitm) // BATCH_SIZE)
    )

    print(f"\nFinished fine-tuning IITM. Best model saved to {args.iitm_output_model}")
    return args.iitm_output_model

# --- Helper and Main Execution (Unchanged) ---
def check_gpu():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"--- Using GPU: {physical_devices[0].name} ---")
            return True
        except RuntimeError as e: print(f"Error setting memory growth: {e}"); return True
    else: print("--- No GPU found, using CPU ---"); return False

def main():
    parser = argparse.ArgumentParser(description="Run two-stage emotion model training (FER -> IITM using filenames)")
    parser.add_argument('--fer_csv', default='data/fer2013.csv', help='Path to fer2013.csv file')
    # Updated help text slightly for clarity
    parser.add_argument('--iitm_dir', default='data/IIITM_Face_Emotion_Dataset', help='Path to the root directory containing IITM images (can be in subfolders)')
    parser.add_argument('--fer_output_model', default='models/fer_trained_model.h5', help='Path to save the best FER-trained model')
    parser.add_argument('--iitm_output_model', default='models/iitm_finetuned_model.h5', help='Path to save the final fine-tuned IITM model')
    parser.add_argument('--epochs_fer', type=int, default=30, help='Epochs for FER2013 training')
    parser.add_argument('--epochs_iitm', type=int, default=20, help='Epochs for IITM fine-tuning')
    parser.add_argument('--lr_fer', type=float, default=1e-3, help='Learning rate for FER training')
    parser.add_argument('--lr_iitm', type=float, default=1e-4, help='Learning rate for IITM fine-tuning')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.fer_output_model), exist_ok=True)
    os.makedirs(os.path.dirname(args.iitm_output_model), exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    gpu_available = check_gpu()

    fer_model_path = train_fer_model(args)

    if fer_model_path and os.path.exists(fer_model_path):
        final_model_path = finetune_iitm_model(args, fer_model_path)
        if final_model_path and os.path.exists(final_model_path):
            print("\n" + "="*30); print(" Training Complete Successfully!"); print("="*30 + "\n")
            print(f"Initial FER model: {fer_model_path}")
            print(f"Final IITM model: {final_model_path}")
            print("\nNext steps:")
            print(f"1. Update demo scripts to use '{final_model_path}'.")
            # Updated to 6 classes
            print("2. Use the 6 IITM emotion labels (NE, SA, SM, SO, SU, YN) for interpretation.")
            print("3. Ensure demo preprocessing matches (48x48 grayscale).")
        else: print("\n--- IITM Fine-tuning Failed ---")
    else: print("\n--- FER Training Failed. Skipping IITM fine-tuning. ---")

if __name__ == '__main__':
    main()
