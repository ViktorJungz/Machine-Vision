import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
model = tf.keras.models.load_model("screw_classifier_model_cropped.h5")

# Define class names
class_names = ['LagBolt', 'SMS', 'SocketScrew', 'WoodScrew']

def preprocess_frame(frame):
    """Resize and normalize frame for model input."""
    img = cv2.resize(frame, (224, 224))  # Resize to match training size
    img = img / 255.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def detect_screws(frame):
    """Find multiple screws in the frame and return bounding boxes."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Reduce noise
    edges = cv2.Canny(blurred, 30, 100)  # Adjusted edge detection thresholds

    # Strengthen edges to improve detection
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    for contour in contours:
        if cv2.contourArea(contour) > 2000:  # Reduced threshold for better detection
            hull = cv2.convexHull(contour)  # Get convex hull for better shape
            x, y, w, h = cv2.boundingRect(hull)  # Get bounding box
            bounding_boxes.append((x, y, w, h))

    return bounding_boxes

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect screws (bounding boxes)
    bounding_boxes = detect_screws(frame)

    for (x, y, w, h) in bounding_boxes:
        # Expand bounding box slightly to capture full screw
        padding = 10  # Adjust if needed
        x, y = max(x - padding, 0), max(y - padding, 0)
        w, h = min(w + 2 * padding, frame.shape[1] - x), min(h + 2 * padding, frame.shape[0] - y)

        # Crop each detected screw
        screw_img = frame[y:y+h, x:x+w]
        screw_img = preprocess_frame(screw_img)

        # Predict screw type
        predictions = model.predict(screw_img)
        class_index = np.argmax(predictions)  # Get highest probability class
        class_name = class_names[class_index]  # Get class label

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Ensure label appears above the box
        label_y = max(y - 10, 20)
        cv2.putText(frame, class_name, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Show the output
    cv2.imshow("Live Multi-Screw Detection", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()