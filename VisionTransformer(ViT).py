import tensorflow as tf
from tensorflow import keras
import keras_cv
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Parameters
img_size = (224, 224)
batch_size = 32
epochs = 10
data_dir = "Pictures"

# Data augmentation
datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=180,
    horizontal_flip=True,
    vertical_flip=True
)

def detect_and_crop_screw(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = image[y:y+h, x:x+w]
        h, w, _ = cropped.shape
        scale = min(img_size[0] / w, img_size[1] / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(cropped, (new_w, new_h))
        padded = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
        start_x = (img_size[0] - new_w) // 2
        start_y = (img_size[1] - new_h) // 2
        padded[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        return padded

    return np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255

def preprocess_image(file_path):
    image = cv2.imread(file_path)
    cropped = detect_and_crop_screw(image)
    cropped = cropped / 255.0
    return cropped

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

X, y, class_names = load_dataset()
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_augmented = datagen.flow(X_train, y_train, batch_size=batch_size)

# Load ViT backbone
vit_model = keras_cv.models.VisionTransformer.from_preset(
    "vit_b16_classification_imagenet1k",
    num_classes=len(class_names)
)

vit_model.compile(
    optimizer=keras.optimizers.Adam(1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

vit_model.fit(X_train_augmented, epochs=epochs, validation_data=(X_val, y_val))
vit_model.save("screw_classifier_vit.h5")
print("Saved as screw_classifier_vit.h5")

# Evaluation
y_pred = vit_model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

conf_matrix = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("ViT Confusion Matrix")
plt.show()
