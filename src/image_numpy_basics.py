import cv2
import numpy as np
import os

INPUT_IMAGE = "datasets/sample.jpg"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(INPUT_IMAGE)
if img is None:
    raise FileNotFoundError("Image not found")

print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("Min pixel:", img.min())
print("Max pixel:", img.max())

resized = cv2.resize(img, (224, 224))
normalized = resized / 255.0
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

cv2.imwrite(f"{OUTPUT_DIR}/resized.jpg", resized)
cv2.imwrite(f"{OUTPUT_DIR}/gray.jpg", gray)
