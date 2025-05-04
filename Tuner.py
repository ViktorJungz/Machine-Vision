import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import keras_tuner as kt
from tensorflow import keras
from sklearn.model_selection import train_test_split

# Set parameters
img_size = (224, 224)
batch_size = 16
data_dir = "Pictures"

# Function to preprocess dataset images
def preprocess_image(file_path):
    image = cv2.imread(file_path)
    resized = cv2.resize(image, img_size)
    normalized = resized / 255.0
    return normalized

# Load dataset and preprocess images
def load_dataset():
    images, labels = [], []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    for class_index, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)

        for img_name in os.listdir(class_path):
            if img_name.startswith('.') or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            img_path = os.path.join(class_path, img_name)
            img = preprocess_image(img_path)
            images.append(img)
            labels.append(class_index)

    return np.array(images), np.array(labels), class_names

# Load and split data
X, y, class_names = load_dataset()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Model-building function for Keras Tuner
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(hp.Choice('conv1_filters', [32, 64, 128]), (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(keras.layers.MaxPooling2D(2, 2))
    model.add(keras.layers.Conv2D(hp.Choice('conv2_filters', [64, 128, 256]), (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(2, 2))
    model.add(keras.layers.Conv2D(hp.Choice('conv3_filters', [128, 256, 512]), (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D(2, 2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu'))
    model.add(keras.layers.Dropout(hp.Float('dropout_rate', 0.3, 0.6, step=0.1)))
    model.add(keras.layers.Dense(len(class_names), activation='softmax'))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])

    return model

# Setup tuner

tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='kt_dir',
    project_name='screw_cnn_tuning'
)

# Early stopping
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Run tuner
tuner.search(X_train, y_train,
             epochs=10,
             validation_data=(X_val, y_val),
             callbacks=[early_stopping])

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(1)[0]

# Print the best hyperparameters
print("Best Hyperparameters:")
print(f"Conv1 Filters: {best_hps.get('conv1_filters')}")
print(f"Conv2 Filters: {best_hps.get('conv2_filters')}")
print(f"Conv3 Filters: {best_hps.get('conv3_filters')}")
print(f"Dense Units: {best_hps.get('dense_units')}")
print(f"Dropout Rate: {best_hps.get('dropout_rate')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")