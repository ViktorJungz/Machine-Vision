import cv2
import os

# Create a directory for saving photos if it doesn't exist
save_dir = "Pictures/WoodScrew"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# Initialize the webcam
cap = cv2.VideoCapture(0)

print("Press the spacebar to take a picture. Press 'q' to quit.")

# Initialize a counter for picture filenames
counter = 151

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the frame
    cv2.imshow('Camera', frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):  # Spacebar to take a picture
        # Generate a unique filename using the counter
        filename = os.path.join(save_dir, f"WoodScrew{counter}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Picture saved as {filename}")
        counter += 1  # Increment the counter
    elif key == ord('q'):  # 'q' to quit
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()