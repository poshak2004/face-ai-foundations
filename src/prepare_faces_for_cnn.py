import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

# ----------------------------
# CONFIG
# ----------------------------
RAW_DIR = "datasets/faces/raw"
PROCESSED_DIR = "datasets/faces/processed"
IMG_SIZE = 96

TRAIN_DIR = os.path.join(PROCESSED_DIR, "train")
VAL_DIR = os.path.join(PROCESSED_DIR, "val")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

# ----------------------------
# LOAD & PREPROCESS
# ----------------------------
images = []
filenames = []

for file in os.listdir(RAW_DIR):
    if not file.lower().endswith(".jpg"):
        continue

    img_path = os.path.join(RAW_DIR, file)
    img = cv2.imread(img_path)

    if img is None:
        continue

    # Resize
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Normalize
    img = img / 255.0

    images.append(img)
    filenames.append(file)

images = np.array(images, dtype="float32")

print("Total images:", len(images))
print("Image shape:", images.shape)

# ----------------------------
# Train / Validation Split
# ----------------------------
train_imgs, val_imgs, train_names, val_names = train_test_split(
    images,
    filenames,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Save processed images
# ----------------------------
def save_images(img_array, name_list, folder):
    for img, name in zip(img_array, name_list):
        out_path = os.path.join(folder, name)
        cv2.imwrite(out_path, (img * 255).astype("uint8"))

save_images(train_imgs, train_names, TRAIN_DIR)
save_images(val_imgs, val_names, VAL_DIR)

print("Train images:", len(train_imgs))
print("Validation images:", len(val_imgs))
print("Processed dataset saved")
