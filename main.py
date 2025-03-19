import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Set parameters
img_size = (224, 224)
batch_size = 32
epochs = 25
data_dir = "Pictures"

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
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs)

# Save the model
model.save("screw_classifier_model_cropped.h5")
print("Model trained and saved as 'screw_classifier_model_cropped.h5'")

# Generate predictions on validation set
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels

# Compute confusion matrix
conf_matrix = confusion_matrix(y_val, y_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
