import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# ----------------------------
# LOAD DATA
# ----------------------------
X_train = np.load("datasets/emotion_processed/X_train.npy")
X_val = np.load("datasets/emotion_processed/X_val.npy")
y_train = np.load("datasets/emotion_processed/y_train.npy")
y_val = np.load("datasets/emotion_processed/y_val.npy")

print("Train:", X_train.shape)
print("Val:", X_val.shape)

# ----------------------------
# MODEL
# ----------------------------
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation="relu", input_shape=(96,96,1)),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation="relu"),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(5, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# TRAIN
# ----------------------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32
)

model.save("models/emotion_cnn.keras")
print("✅ Emotion CNN trained and saved")
