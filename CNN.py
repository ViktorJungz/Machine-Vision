import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from keras.src.callbacks import EarlyStopping
from sklearn.metrics import precision_recall_curve, average_precision_score

# Set parameters
img_size = (224, 224)
batch_size = 16
epochs = 100
data_dir = "Pictures"

# Define EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights=True, patience=5)

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=180,  # Randomly rotate images
    horizontal_flip=True,
    vertical_flip=True  # Ensures different orientations are seen
)

# Function to resize and normalize the image
def resize_and_normalize_image(image):
    """Resize the image and normalize pixel values."""
    resized = cv2.resize(image, img_size)  # Resize the image
    normalized = resized / 255.0  # Normalize pixel values
    return normalized

# Function to preprocess dataset images
def preprocess_image(file_path):
    """Loads and preprocesses an image."""
    image = cv2.imread(file_path)
    processed_image = resize_and_normalize_image(image)  # Resize and normalize
    return processed_image

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
X_train = datagen.flow(X_train, y_train, batch_size=batch_size)

# Build the model
model = keras.Sequential([
    keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(192, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(len(class_names), activation='softmax')
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),    # Reudced from 0.001 to 0.0001
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model with the callback
history = model.fit(X_train, epochs=epochs, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Save the model
model.save("screw_classifier_model_resized.h5")
print("Model trained and saved as 'screw_classifier_model_resized.h5'")

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

# Plot training and validation loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Add labels and legend
# Compute Precision-Recall curve for each class
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(class_names):
    # Get true binary labels for the current class
    y_true_binary = (y_val == i).astype(int)
    # Get predicted probabilities for the current class
    y_pred_prob = y_pred[:, i]
    # Compute precision and recall
    precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_prob)
# Plot the curve
plt.plot(recall, precision, label=f'{class_name} (AP={average_precision_score(y_true_binary, y_pred_prob):.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='lower left')
plt.grid(True)
plt.show()
