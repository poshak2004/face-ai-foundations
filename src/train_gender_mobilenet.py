import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ----------------------------
# CONFIG
# ----------------------------
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 8
DATASET_DIR = "datasets/age_gender"

# ----------------------------
# DATA GENERATOR (WITH AUGMENTATION)
# ----------------------------
datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    validation_split=0.2,
    rotation_range=15,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1,
    horizontal_flip=True
)

train_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="training"
)

val_gen = datagen.flow_from_directory(
    DATASET_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    subset="validation"
)

# ----------------------------
# BASE MODEL
# ----------------------------
base_model = MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)

# 🔥 PARTIAL FINE-TUNING
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# ----------------------------
# MODEL
# ----------------------------
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ----------------------------
# TRAIN
# ----------------------------
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ----------------------------
# SAVE (KERAS 3 SAFE)
# ----------------------------
os.makedirs("models", exist_ok=True)
model.save("models/gender_mobilenet.keras")

print("✅ Gender model trained and saved (improved)")
