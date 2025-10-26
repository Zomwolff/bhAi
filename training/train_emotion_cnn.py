# File: models_updated.py
# Based on models.py from the PDF [cite: 392-431]
# Modifications:
# - Changed default num_classes to handle both FER (7) and IITM (5) during different stages.
# - Input shape remains (48, 48, 1) as used in both training stages.

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, BatchNormalization,
                                     Activation, MaxPooling2D, Dropout,
                                     Flatten, Dense)

def build_emotion_cnn(input_shape=(48, 48, 1), num_classes=7): # Default to 7 for initial FER training
    """Builds the CNN model architecture."""
    inp = Input(shape=input_shape)

    # Block 1
    x = Conv2D(32, (3, 3), padding='same')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 2
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.25)(x)

    # Block 3
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # Removed one Conv layer from original PDF Block 3 for slight simplification
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.3)(x)

    # Flatten and Dense layers
    x = Flatten()(x)
    x = Dense(1024)(x) # Kept the dense layer size
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.5)(x)

    out = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=inp, outputs=out)
    return model

# You can test if the model builds:
# if __name__ == '__main__':
#     model_fer = build_emotion_cnn(num_classes=7)
#     model_fer.summary()
#     print("\nBuilding model for 5 classes (like IITM):")
#     model_iitm = build_emotion_cnn(num_classes=5)
#     model_iitm.summary()
