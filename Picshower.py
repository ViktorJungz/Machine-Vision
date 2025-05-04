import cv2
import numpy as np

# === Improved detect_bbox function ===
def detect_bbox(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Lower thresholds to capture more detail
    edges = cv2.Canny(blurred, 30, 100)

    # Dilate edges to connect broken parts of the screw
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Combine all large contours into one bounding box
        screw_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 500]
        if screw_contours:
            x, y, w, h = cv2.boundingRect(np.vstack(screw_contours))
            return (x, y, x + w, y + h)
    return None

# === Load and process an image ===
image_path = "Pictures\\LagBolt\\WoodScrew488.jpg"  # <-- Change this to your test image path
image = cv2.imread(image_path)

if image is None:
    print("❌ Failed to load image.")
else:
    bbox = detect_bbox(image)

    if bbox:
        # Draw bounding box
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(image, "Detected", (bbox[0], bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print("✅ Bounding box drawn.")
    else:
        print("⚠️ No object detected.")

    # Show image
    cv2.imshow("Detected Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()