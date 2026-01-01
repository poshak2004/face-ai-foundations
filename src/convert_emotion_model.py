import keras
from keras import layers

OLD_MODEL_PATH = "models/emotion_cnn"
NEW_MODEL_PATH = "models/emotion_cnn.keras"

# Load SavedModel as inference-only layer
emotion_layer = layers.TFSMLayer(
    OLD_MODEL_PATH,
    call_endpoint="serving_default"
)

# Wrap in a Keras Model
inputs = keras.Input(shape=(96, 96, 1))
outputs = emotion_layer(inputs)
model = keras.Model(inputs, outputs)

# Save in new Keras v3 format
model.save(NEW_MODEL_PATH)

print("✅ Model successfully converted to emotion_cnn.keras")
