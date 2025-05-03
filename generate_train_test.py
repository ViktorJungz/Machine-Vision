import os
import random

dataset_dir = "VOC_dataset"  # Path to your VOC dataset
image_dir = os.path.join(dataset_dir, "JPEGImages")
image_files = [f.replace(".jpg", "").replace(".png", "").replace(".jpeg", "") for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
random.shuffle(image_files)

# Split 80% for training, 20% for validation
split_ratio = 0.8
train_files = image_files[:int(len(image_files) * split_ratio)]
val_files = image_files[int(len(image_files) * split_ratio):]

# Write to train.txt
with open(os.path.join(dataset_dir, "ImageSets", "Main", "train.txt"), "w") as f:
    for file in train_files:
        f.write(file + "\n")

# Write to val.txt
with open(os.path.join(dataset_dir, "ImageSets", "Main", "val.txt"), "w") as f:
    for file in val_files:
        f.write(file + "\n")

print("✅ train.txt and val.txt generated.")
