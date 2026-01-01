import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = 96
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = "datasets/faces/processed/train"
VAL_DIR = "datasets/faces/processed/val"

# ----------------------------
# LOAD IMAGES
# ----------------------------
def load_images(folder):
    images = []
    labels = []

    for file in os.listdir(folder):
        if not file.lower().endswith(".jpg"):
            continue

        img_path = os.path.join(folder, file)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img / 255.0

        images.append(img)
        labels.append(1)  # dummy label

    return np.array(images, dtype="float32"), np.array(labels)

X_train, y_train = load_images(TRAIN_DIR)
X_val, y_val = load_images(VAL_DIR)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)

# ----------------------------
# CNN MODEL
# ----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# TRAIN
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# ----------------------------
# SAVE MODEL
# ----------------------------
os.makedirs("models", exist_ok=True)
model.save("models/cnn_baseline_face_model")

print("✅ Model training complete and saved")

