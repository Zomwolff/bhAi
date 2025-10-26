# File: resume_training.py
# Loads the previously trained emotion model and continues training for more epochs.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import collections
import tensorflow as tf

# Importing Deep Learning Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model # Use load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.utils import class_weight
import datetime

# --- Configuration ---
PICTURE_SIZE = 48
DATA_DIR = "smth" # Ensure this path is correct
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "test")
BATCH_SIZE = 256 # Keep consistent with the previous training session for comparable steps/epoch
NO_OF_CLASSES = 7

# --- IMPORTANT: Specify paths and epochs ---
EXISTING_MODEL_PATH = "models/notebook_emotion_cnn_weighted_1hr_opt.h5" # Model to load
NEW_MODEL_SAVE_PATH = "models/notebook_emotion_cnn_weighted_resumed.h5" # Where to save the improved model
ADDITIONAL_EPOCHS = 30 # How many *more* epochs to train (e.g., train up to epoch 10 + 30 = 40)
INITIAL_EPOCH = 10 # Start counting epochs from here (matches the previous run's end)
NEW_LEARNING_RATE = 5e-5 # Use a smaller learning rate for fine-tuning (e.g., 0.00005)

# --- Enable Mixed Precision (Keep consistent with previous training) ---
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("--- GPU detected. Enabled Mixed Precision. ---")
    except Exception as e:
        print(f"--- GPU detected but could not enable Mixed Precision: {e}. ---")
else:
    print("--- No GPU detected, running on CPU. ---")

# --- 1. Data Generators (Same as before) ---
print("Setting up Data Generators...")
datagen_train = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1,
                                   height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
                                   horizontal_flip=True, fill_mode='nearest')
datagen_val = ImageDataGenerator(rescale=1./255)

try:
    train_set = datagen_train.flow_from_directory(TRAIN_DIR, target_size=(PICTURE_SIZE, PICTURE_SIZE),
                                                  color_mode="grayscale", batch_size=BATCH_SIZE,
                                                  class_mode='categorical', shuffle=True)
    test_set = datagen_val.flow_from_directory(VALIDATION_DIR, target_size=(PICTURE_SIZE, PICTURE_SIZE),
                                               color_mode="grayscale", batch_size=BATCH_SIZE,
                                               class_mode='categorical', shuffle=False)
except FileNotFoundError as e:
    print(f"\nError: Data directory not found. Check DATA_DIR path ('{DATA_DIR}'). Details: {e}")
    exit()

if train_set.num_classes != NO_OF_CLASSES:
    print(f"Warning: Class number mismatch. Expected {NO_OF_CLASSES}, found {train_set.num_classes}")
    NO_OF_CLASSES = train_set.num_classes

# --- 2. Calculate Class Weights (Same as before) ---
print("\nCalculating class weights...")
counter = collections.Counter(train_set.classes)
class_labels_map = {v: k for k, v in train_set.class_indices.items()}
class_weights_calculated = class_weight.compute_class_weight(class_weight='balanced',
                                                             classes=np.unique(train_set.classes),
                                                             y=train_set.classes)
class_weights_dict = dict(enumerate(class_weights_calculated))
print("Using Class Weights:")
for i in range(NO_OF_CLASSES): print(f"  Class {i} ({class_labels_map.get(i)}): Weight = {class_weights_dict.get(i):.3f}")

# --- 3. Load Existing Model ---
print(f"\nLoading existing model from: {EXISTING_MODEL_PATH}")
if not os.path.exists(EXISTING_MODEL_PATH):
    print(f"Error: Model file not found at {EXISTING_MODEL_PATH}. Cannot resume training.")
    exit()
try:
    model = load_model(EXISTING_MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# --- 4. Re-compile with potentially lower learning rate ---
print(f"Re-compiling model with Adam optimizer and learning rate: {NEW_LEARNING_RATE}")
opt = Adam(learning_rate=NEW_LEARNING_RATE)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary() # Show summary again

# --- 5. Callbacks (Adjust paths and patience if needed) ---
print("\nSetting up Callbacks for resumed training...")
os.makedirs(os.path.dirname(NEW_MODEL_SAVE_PATH), exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Save the best *resumed* model to a new file
checkpoint = ModelCheckpoint(NEW_MODEL_SAVE_PATH, monitor='val_accuracy', verbose=1,
                             save_best_only=True, mode='max')

# Early stopping (can adjust patience)
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, # Maybe increase patience a bit
                               verbose=1, restore_best_weights=True)

# Reduce learning rate (can adjust patience)
reduce_learningrate = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5,
                                        verbose=1, min_delta=0.0001)

# TensorBoard logs for the resumed session
log_dir = os.path.join("logs", "resume_notebook_weighted", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks_list = [early_stopping, checkpoint, reduce_learningrate, tensorboard_callback]

# --- 6. Resume Training ---
print(f"\nResuming training for {ADDITIONAL_EPOCHS} additional epochs (starting from epoch {INITIAL_EPOCH})...")
print(f"Total target epochs: {INITIAL_EPOCH + ADDITIONAL_EPOCHS}")
print(f"TensorBoard logs: tensorboard --logdir {log_dir}")

steps_per_epoch = train_set.samples // BATCH_SIZE
validation_steps = test_set.samples // BATCH_SIZE
if steps_per_epoch == 0 or validation_steps == 0:
    raise ValueError("Steps per epoch or validation steps is zero.")

try:
    history = model.fit(
        train_set,
        steps_per_epoch=steps_per_epoch,
        epochs=INITIAL_EPOCH + ADDITIONAL_EPOCHS, # Total number of epochs desired
        initial_epoch=INITIAL_EPOCH, # Tell Keras where we left off
        validation_data=test_set,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        class_weight=class_weights_dict
    )
except tf.errors.ResourceExhaustedError:
    print("\nError: GPU out of memory (OOM). Try reducing BATCH_SIZE.")
    exit()
except Exception as e:
    print(f"\nAn error occurred during training: {e}")
    exit()

print(f"\nTraining finished. Best resumed model saved to {NEW_MODEL_SAVE_PATH}")

# --- 7. Plotting Accuracy & Loss (Optional) ---
# Note: This plot will only show the history for the *resumed* session.
# For a full history, you'd need to load and combine history logs.
print("\nPlotting resumed training history...")
plt.style.use('dark_background')
plt.figure(figsize=(20, 10))
# ... (plotting code similar to the previous script) ...
plt.savefig("resumed_training_history.png")
print("Saved resumed training history plot to resumed_training_history.png")
