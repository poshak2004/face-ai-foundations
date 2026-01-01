print(">>> prepare_fer2013_dataset.py started")

import pandas as pd
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split

# ----------------------------
# CONFIG
# ----------------------------
CSV_PATH = "datasets/fer2013/fer2013.csv"
IMG_SIZE = 96

# Keep only 5 emotions
LABEL_MAP = {
    0: 0,  # Angry
    3: 1,  # Happy
    4: 2,  # Sad
    5: 3,  # Surprise
    6: 4   # Neutral
}

print("Checking CSV path:", CSV_PATH)
assert os.path.exists(CSV_PATH), "❌ CSV FILE NOT FOUND"

# ----------------------------
# LOAD CSV
# ----------------------------
df = pd.read_csv(CSV_PATH)
print("Total rows in CSV:", len(df))
print("Columns:", df.columns)

X = []
y = []

# ----------------------------
# PROCESS ROWS
# ----------------------------
for i, row in df.iterrows():
    emotion = row["emotion"]
    if emotion not in LABEL_MAP:
        continue

    pixels = np.fromstring(row["pixels"], sep=" ", dtype="uint8")
    if pixels.size != 48 * 48:
        continue

    img = pixels.reshape(48, 48)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0

    X.append(img)
    y.append(LABEL_MAP[emotion])

    if i % 5000 == 0:
        print("Processed rows:", i)

print("Total images kept:", len(X))
assert len(X) > 0, "❌ NO IMAGES PROCESSED"

X = np.array(X, dtype="float32")
X = np.expand_dims(X, -1)
y = np.array(y)

# ----------------------------
# SPLIT
# ----------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ----------------------------
# SAVE
# ----------------------------
os.makedirs("datasets/emotion_processed", exist_ok=True)

np.save("datasets/emotion_processed/X_train.npy", X_train)
np.save("datasets/emotion_processed/X_val.npy", X_val)
np.save("datasets/emotion_processed/y_train.npy", y_train)
np.save("datasets/emotion_processed/y_val.npy", y_val)

print("✅ FER-2013 DATASET SAVED SUCCESSFULLY")
