# File: train_notebook_model_weighted.py
# Adapts the emotion classification notebook, adding class weights for normalization.
# Optimized for a target training time of approximately 1 hour by reducing epochs,
# enabling mixed precision, increasing batch size, and simplifying the model slightly.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
import collections # For counting class distribution
import tensorflow as tf # Import tensorflow

# Importing Deep Learning Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Dense, Input, Dropout, GlobalAveragePooling2D, Flatten,
                                     Conv2D, BatchNormalization, Activation, MaxPooling2D)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from sklearn.utils import class_weight # Import for calculating class weights
import datetime

# --- Configuration ---
PICTURE_SIZE = 48 # Input image size (48x48 grayscale)
# IMPORTANT: Update this path to your actual dataset location
DATA_DIR = "smth" # Contains 'train' and 'validation' subfolders
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VALIDATION_DIR = os.path.join(DATA_DIR, "test")
BATCH_SIZE = 256 # Increased from 128 (Reduce if OOM error occurs)
NO_OF_CLASSES = 7 # From notebook (FER dataset)
EPOCHS = 10 # Drastically reduced from 48 for ~1hr target based on 11.5 min/epoch baseline
MODEL_SAVE_PATH = "models/notebook_emotion_cnn_weighted_1hr_opt.h5" # Updated save path

# --- Enable Mixed Precision (for speed on compatible GPUs) ---
# Check if GPU is available and supports mixed precision
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        # Set memory growth for GPUs
        for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Enable mixed precision
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("--- GPU detected. Enabled Mixed Precision (mixed_float16) for potential speedup. ---")
    except Exception as e:
        print(f"--- GPU detected but could not enable Mixed Precision: {e}. Check TF version and GPU compatibility. ---")
else:
    print("--- No GPU detected, running on CPU. Mixed precision not applied. ---")


# --- 1. Data Generators ---
print(f"Setting up Data Generators for folders:\n Training: {TRAIN_DIR}\n Validation: {VALIDATION_DIR}")

# Training data generator with augmentation
datagen_train = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Validation data generator (only rescale)
datagen_val = ImageDataGenerator(rescale=1./255)

try:
    train_set = datagen_train.flow_from_directory(
        TRAIN_DIR,
        target_size=(PICTURE_SIZE, PICTURE_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    test_set = datagen_val.flow_from_directory(
        VALIDATION_DIR,
        target_size=(PICTURE_SIZE, PICTURE_SIZE),
        color_mode="grayscale",
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False # Important for validation/testing
    )
except FileNotFoundError as e:
    print(f"\nError: Data directory not found. Please check the DATA_DIR path ('{DATA_DIR}') and ensure it contains 'train' and 'validation' subfolders.")
    print(f"Details: {e}")
    exit() # Exit if data is not found

# Verify classes
if train_set.num_classes != NO_OF_CLASSES:
    print(f"Warning: Expected {NO_OF_CLASSES} classes but found {train_set.num_classes} in {TRAIN_DIR}")
    print("Class indices found:", train_set.class_indices)
    NO_OF_CLASSES = train_set.num_classes # Adjust if necessary

# --- 2. Calculate Class Weights ---
print("\nCalculating class weights...")
# Count samples per class using the generator's internal counts
counter = collections.Counter(train_set.classes)
total_samples = sum(counter.values())
print("Training Class Distribution:")
class_labels_map = {v: k for k, v in train_set.class_indices.items()} # Map index back to name
for i in range(NO_OF_CLASSES):
    label_name = class_labels_map.get(i, f"Unknown_Idx_{i}")
    count = counter.get(i, 0)
    print(f"  Class {i} ({label_name}): {count} samples")

# Calculate weights using sklearn
class_weights_calculated = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_set.classes), # Use only the classes present (0-6)
    y=train_set.classes # Use the numerical labels from the generator
)
# Convert to dictionary format for Keras
class_weights_dict = dict(enumerate(class_weights_calculated))
print("Calculated Class Weights:")
for i in range(NO_OF_CLASSES):
    label_name = class_labels_map.get(i, f"Unknown_Idx_{i}")
    weight = class_weights_dict.get(i, "N/A")
    print(f"  Class {i} ({label_name}): Weight = {weight:.3f}")


# --- 3. Model Building (Slightly Simplified) ---
print("\nBuilding the simplified CNN model...")
model = Sequential()

# 1st CNN layer
model.add(Conv2D(64, (3, 3), padding='same', input_shape=(PICTURE_SIZE, PICTURE_SIZE, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 2nd CNN layer
model.add(Conv2D(128, (5, 5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 3rd CNN layer
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# 4th CNN layer
model.add(Conv2D(512, (3, 3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

# Fully connected 1st layer (kept)
model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

# Fully connected layer 2nd layer (REMOVED Dense(512) block for simplification)
# model.add(Dense(512))
# model.add(BatchNormalization())
# model.add(Activation('relu'))
# model.add(Dropout(0.25))

# Output Layer
model.add(Dense(NO_OF_CLASSES, activation='softmax', dtype='float32')) # Ensure output is float32 when using mixed precision

# Compile the model
opt = Adam(learning_rate=0.0001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# --- 4. Callbacks (Similar to Notebook) ---
print("\nSetting up Callbacks...")
# Ensure models directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
os.makedirs("logs", exist_ok=True) # For TensorBoard

checkpoint = ModelCheckpoint(
    MODEL_SAVE_PATH,
    monitor='val_accuracy', # Monitor validation accuracy
    verbose=1,
    save_best_only=True,
    mode='max' # Save the model with max validation accuracy
)

# Early stopping if val_loss doesn't improve
# Reduce patience since we have fewer epochs
early_stopping = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5, # Reduced patience
    verbose=1,
    restore_best_weights=True
)

# Reduce learning rate if val_loss plateaus
# Reduce patience since we have fewer epochs
reduce_learningrate = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=3, # Reduced patience
    verbose=1,
    min_delta=0.0001
)

# TensorBoard for visualization
log_dir = os.path.join("logs", "fit_notebook_weighted_1hr", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks_list = [early_stopping, checkpoint, reduce_learningrate, tensorboard_callback]

# --- 5. Training the Model (with Class Weights) ---
print(f"\nStarting training for {EPOCHS} epochs...")
print(f"Targeting completion within ~1 hour.")
print(f"TensorBoard logs: tensorboard --logdir {log_dir}")

# Calculate steps per epoch
steps_per_epoch = train_set.samples // BATCH_SIZE
validation_steps = test_set.samples // BATCH_SIZE

if steps_per_epoch == 0 or validation_steps == 0:
    raise ValueError("Steps per epoch or validation steps is zero. Check batch size and dataset size/paths.")

# Use model.fit
try:
    history = model.fit(
        train_set,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=test_set,
        validation_steps=validation_steps,
        callbacks=callbacks_list,
        class_weight=class_weights_dict # Apply class weights
    )
except tf.errors.ResourceExhaustedError:
    print("\nError: GPU out of memory (OOM).")
    print(f"Try reducing the BATCH_SIZE (currently {BATCH_SIZE}) in the script and run again.")
    exit()
except Exception as e:
    print(f"\nAn error occurred during training: {e}")
    exit()


print(f"\nTraining finished. Best model saved to {MODEL_SAVE_PATH}")

# --- 6. Plotting Accuracy & Loss (Optional) ---
print("\nPlotting training history...")
plt.style.use('dark_background')

plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam (Weighted Training, 1hr Target)', fontsize=10)
plt.ylabel('Loss', fontsize=16)
if 'loss' in history.history:
    plt.plot(history.history['loss'], label='Training Loss')
if 'val_loss' in history.history:
    plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
if 'accuracy' in history.history:
    plt.plot(history.history['accuracy'], label='Training Accuracy')
if 'val_accuracy' in history.history:
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')

plt.savefig("training_history_weighted_1hr.png") # Save the plot
print("Saved training history plot to training_history_weighted_1hr.png")
# plt.show() # Uncomment if running interactively