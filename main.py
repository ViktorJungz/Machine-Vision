import tensorflow as tf
from tensorflow import keras
layers = keras.layers
import matplotlib.pyplot as plt
import os

# Set parameters
img_size = (224, 224)
batch_size = 32
epochs = 10  # Adjust as needed
data_dir = "Pictures"

# Load training dataset (80%)
train_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # 20% for validation, 80% for training
    subset="training",  # Use the training split
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

# Load validation dataset (20%)
val_ds = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # 20% for validation, 80% for training
    subset="validation",  # Use the validation split
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)

# Extract class names before preprocessing
class_names = train_ds.class_names
num_classes = len(class_names)

# Normalize pixel values
normalization_layer = layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# Build the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Plot training results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save the model
model.save("screw_classifier_model.h5")

print("Model training complete and saved as 'screw_classifier_model.h5'")