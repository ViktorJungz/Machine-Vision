import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Set parameters
img_size = (224, 224)
image_path = "Pictures\LagBolt\LagBolt75.jpg"  # Change this to your test image

def visualize_preprocessing(image_path):
    image = cv2.imread(image_path)
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
    else:
        padded = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255
    
    images = [image, gray, blurred, edges, cropped, padded]
    titles = ["Original", "Grayscale", "Blurred", "Edges", "Cropped", "Final Processed"]
    
    plt.figure(figsize=(12, 8))
    for i in range(len(images)):
        plt.subplot(2, 3, i+1)
        cmap = 'gray' if len(images[i].shape) == 2 else None
        plt.imshow(images[i], cmap=cmap)
        plt.title(titles[i])
        plt.axis("off")
    plt.tight_layout()
    plt.show()

# Run visualization
visualize_preprocessing(image_path)