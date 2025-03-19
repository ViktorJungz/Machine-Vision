import cv2
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
import matplotlib.pyplot as plt
import os

# Set Parameters
img_size = (224, 224) # Image size
batch_size = 32 # Batch size
epochs = 10 # Number of epochs (iterations the model goes through the training data)
data_dir = 'dataset/' # Directory where the data is stored

# Load and preprocess the data
train_dataset, validate_dataset = keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,  # 20% for validation, 80% for training
    subset="both",  # Load both training and validation splits
    seed=42,
    image_size=img_size,
    batch_size=batch_size
)


# Normalize pixelvalues
normalization_layer = layers.Rescaling(1./255) # Rescale pixel values to [0, 1]
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y)) # Apply the normalization layer to the training dataset
validate_dataset = validate_dataset.map(lambda x, y: (normalization_layer(x), y)) # Apply the normalization layer to the validation dataset

# Build the model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)), # Convolutional layer with 32 filters and a kernel size of 3x3
    layers.MaxPooling2D(2, 2), # Max pooling layer with a pool size of 2x2
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(), # Flatten the output of the convolutional layers
    layers.Dense(128, activation='relu'), # Fully connected layer with 128 units
    layers.Dropout(0.5), # Dropout layer with a dropout rate of 0.5
    layers.Dense(len(train_dataset.class_names), activation='softmax') # Output layer with a number of units equal to the number of classes in the dataset
])

# Compile the model
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), # Adam optimizer with a learning rate of 0.001
              loss='sparse_categorical_crossentropy', # Sparse categorical crossentropy loss function
              metrics=['accuracy']) # Accuracy metric

# Train the model
history = model.fit(
    train_dataset, 
    validation_data=validate_dataset,
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

                                                    
