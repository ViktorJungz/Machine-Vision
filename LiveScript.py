import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model("screw_classifier_model.h5")

# Define class names (ensure this matches training class names)
class_names = ['ScrewType1', 'ScrewType2', 'ScrewType3']  # Update based on your dataset

def preprocess_frame(frame):
    """Resize and normalize frame for model input."""
    img = cv2.resize(frame, (224, 224))  # Resize to match training size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess frame
    img = preprocess_frame(frame)
    
    # Make prediction
    predictions = model.predict(img)
    class_index = np.argmax(predictions)  # Get highest probability class
    class_name = class_names[class_index]  # Get class label
    
    # Display result on frame
    cv2.putText(frame, f"Predicted: {class_name}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Screw Detection", frame)
    
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
