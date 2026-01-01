import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# ---------------- CONFIG ----------------
IMAGE_PATH = "datasets/sample.jpg"
OUTPUT_DIR = "outputs"
HAAR_MODEL_PATH = "src/haarcascade_frontalface_default.xml"

# ----------------------------------------
# Create output directory if not exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load image
img = cv2.imread(IMAGE_PATH)
if img is None:
    raise FileNotFoundError("Image not found at datasets/sample.jpg")

print("Original shape:", img.shape)

# ----------------------------------------
# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ----------------------------------------
# Edge Detection (Canny)
edges = cv2.Canny(gray, threshold1=100, threshold2=200)
cv2.imwrite(f"{OUTPUT_DIR}/edges.jpg", edges)
print("Edges saved:", os.path.exists(f"{OUTPUT_DIR}/edges.jpg"))

# ----------------------------------------
# Gaussian Blur (Noise Reduction)
blurred = cv2.GaussianBlur(gray, (7, 7), 0)
cv2.imwrite(f"{OUTPUT_DIR}/blurred.jpg", blurred)

# ----------------------------------------
# Histogram Plot
plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Pixel value")
plt.ylabel("Frequency")
plt.hist(gray.ravel(), bins=256)
plt.savefig(f"{OUTPUT_DIR}/histogram.png")
plt.close()

# ----------------------------------------
# Face Detection using Haar Cascade
face_cascade = cv2.CascadeClassifier(HAAR_MODEL_PATH)

faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5
)

print("Faces detected:", len(faces))

for (x, y, w, h) in faces:
    cv2.rectangle(
        img,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

cv2.imwrite(f"{OUTPUT_DIR}/faces_detected.jpg", img)

print("All outputs saved in /outputs folder")
