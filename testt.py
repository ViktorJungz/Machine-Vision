import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from kerastuner.tuners import RandomSearch



# Set parameters
img_size = (224, 224)
batch_size = 64
epochs = 25
data_dir = "Pictures"

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=180,  # Randomly rotate images
    horizontal_flip=True,
    vertical_flip=True  # Ensures different orientations are seen
)

# Function to detect and crop the screw from an image
def detect_and_crop_screw(image):
    """Detects the screw in an image and crops it before resizing."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 50, 150)  # Edge detection

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)  # Find largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)  # Get bounding box

        # Crop the detected screw
        cropped = image[y:y+h, x:x+w]

        # Resize the cropped screw to match model input size
        cropped = cv2.resize(cropped, img_size)
        return cropped

    # If no screw is found, return the original resized image
    return cv2.resize(image, img_size)

# Function to preprocess dataset images
def preprocess_image(file_path):
    """Loads, detects, and preprocesses an image."""
    image = cv2.imread(file_path)
    cropped = detect_and_crop_screw(image)  # Crop the screw
    cropped = cropped / 255.0  # Normalize pixel values
    return cropped

# Load dataset and preprocess images
def load_dataset():
    images, labels = [], []
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])  # Ignore .DS_Store

    for class_index, class_name in enumerate(class_names):
        class_path = os.path.join(data_dir, class_name)

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            # Skip hidden/system files (like .DS_Store) and non-image files
            if img_name.startswith('.') or not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img = preprocess_image(img_path)
            images.append(img)
            labels.append(class_index)

    # Convert to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels, class_names

# Load the dataset
X, y, class_names = load_dataset()

# Split dataset into training (80%) and validation (20%)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply data augmentation to training data
datagen.fit(X_train)  # Fit the data generator to the training data

# Create data generators for training and validation
train_generator = datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = ImageDataGenerator().flow(X_val, y_val, batch_size=batch_size)  # No augmentation for validation data

# Define a function to build the model with tunable hyperparameters
def build_model(hp):
    model = keras.Sequential([
        keras.layers.Conv2D(
            filters=hp.Choice('filters_1', values=[32, 64, 128]),
            kernel_size=hp.Choice('kernel_size_1', values=[3, 5]),
            activation='relu',
            input_shape=(224, 224, 3)
        ),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Conv2D(
            filters=hp.Choice('filters_2', values=[64, 128, 256]),
            kernel_size=hp.Choice('kernel_size_2', values=[3, 5]),
            activation='relu'
        ),
        keras.layers.MaxPooling2D(2, 2),
        keras.layers.Flatten(),
        keras.layers.Dense(
            units=hp.Choice('dense_units', values=[64, 128, 256]),
            activation='relu'
        ),
        keras.layers.Dropout(hp.Choice('dropout_rate', values=[0.3, 0.5, 0.7])),
        keras.layers.Dense(len(class_names), activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Initialize Keras Tuner
tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,  # Number of different hyperparameter combinations to try
    executions_per_trial=2,  # Number of models to train per combination
    directory='hyperparameter_tuning',
    project_name='screw_classifier'
)

# Perform hyperparameter search
tuner.search(train_generator, epochs=10, validation_data=val_generator, steps_per_epoch=len(X_train) // batch_size)

# Get the best hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

# Build the model with the best hyperparameters
model = tuner.hypermodel.build(best_hps)

# Train the model with the best hyperparameters
history = model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Save the optimized model
model.save("optimized_screw_classifier_model.h5")
print("Optimized model trained and saved as 'optimized_screw_classifier_model.h5'")