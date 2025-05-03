import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom.minidom import parseString


# === Your Parameters ===
input_dir = "Pictures"  # Folder with class folders
output_dir = "VOC_dataset"
img_size = (224, 224)

# === Create output folders ===
voc_images = os.path.join(output_dir, "JPEGImages")
voc_annots = os.path.join(output_dir, "Annotations")
os.makedirs(voc_images, exist_ok=True)
os.makedirs(voc_annots, exist_ok=True)

# === Function to detect screw and get bounding box ===
def detect_bbox(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        return (x, y, x + w, y + h)
    return None

# === Function to create VOC annotation XML ===
def create_voc_xml(filename, width, height, bbox, class_name):
    annotation = ET.Element("annotation")

    ET.SubElement(annotation, "folder").text = "JPEGImages"
    ET.SubElement(annotation, "filename").text = filename
    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(width)
    ET.SubElement(size, "height").text = str(height)
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(annotation, "segmented").text = "0"

    obj = ET.SubElement(annotation, "object")
    ET.SubElement(obj, "name").text = class_name
    ET.SubElement(obj, "pose").text = "Unspecified"
    ET.SubElement(obj, "truncated").text = "0"
    ET.SubElement(obj, "difficult").text = "0"
    bbox_tag = ET.SubElement(obj, "bndbox")
    ET.SubElement(bbox_tag, "xmin").text = str(bbox[0])
    ET.SubElement(bbox_tag, "ymin").text = str(bbox[1])
    ET.SubElement(bbox_tag, "xmax").text = str(bbox[2])
    ET.SubElement(bbox_tag, "ymax").text = str(bbox[3])

    return parseString(ET.tostring(annotation)).toprettyxml(indent="  ")

# === Process Images ===
class_names = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])

for class_name in class_names:
    class_path = os.path.join(input_dir, class_name)
    for file_name in os.listdir(class_path):
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(class_path, file_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        bbox = detect_bbox(img)
        if bbox is None:
            continue

        height, width, _ = img.shape

        # Save image to VOC folder
        new_filename = f"{class_name}_{file_name}"
        cv2.imwrite(os.path.join(voc_images, new_filename), img)

        # Save annotation
        xml_content = create_voc_xml(new_filename, width, height, bbox, class_name)
        with open(os.path.join(voc_annots, new_filename.replace(".jpg", ".xml").replace(".png", ".xml")), "w") as f:
            f.write(xml_content)

print("✅ VOC annotation and image generation complete.")

