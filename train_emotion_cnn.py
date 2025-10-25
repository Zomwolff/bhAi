# train_emotion_cnn.py
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.optimizers import Adam
import numpy as np
from models import build_emotion_cnn
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def main(args):
    data = np.load(args.data_npz)
    X_train = data['X_train']
    X_test = data['X_test']
    y_train = data['y_train']
    y_test = data['y_test']
    model = build_emotion_cnn(input_shape=X_train.shape[1:],
    num_classes=y_train.shape[1])
    model.compile(optimizer=Adam(learning_rate=1e-3),
    loss='categorical_crossentropy', metrics=['accuracy'])
    datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
    )
    datagen.fit(X_train)
    callbacks = [
        ModelCheckpoint(args.output, monitor='val_accuracy',
        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6,
        min_lr=1e-6, verbose=1),
        EarlyStopping(monitor='val_loss', patience=12,
        restore_best_weights=True, verbose=1)
    ]
    model.fit(datagen.flow(X_train, y_train, batch_size=64),
    validation_data=(X_test, y_test),
    epochs=args.epochs,
    callbacks=callbacks,
    steps_per_epoch=len(X_train)//64)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_npz', required=True, help='processed data npz (output of data_preprocessing.py)')
    parser.add_argument('--output', default='models/emotion_cnn.h5')
    parser.add_argument('--epochs', type=int, default=60)
    args = parser.parse_args()
    main(args)