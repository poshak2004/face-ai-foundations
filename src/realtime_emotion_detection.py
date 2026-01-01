import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tensorflow as tf

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "models/emotion_cnn.keras"
FACE_MODEL = "models/blaze_face_short_range.tflite"
IMG_SIZE = 96

EMOTIONS = ["Angry", "Happy", "Sad", "Surprise", "Neutral"]

# ----------------------------
# LOAD EMOTION MODEL
# ----------------------------
emotion_model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Emotion CNN loaded")

# ----------------------------
# MEDIAPIPE FACE DETECTOR
# ----------------------------
base_options = python.BaseOptions(model_asset_path=FACE_MODEL)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.FaceDetector.create_from_options(options)

# ----------------------------
# WEBCAM
# ----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Webcam not accessible")

print("🎥 Real-time Emotion Detection started (press 'q' to quit)")

# ----------------------------
# MAIN LOOP
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)

    if result.detections:
        for detection in result.detections:
            bbox = detection.bounding_box
            x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            face = frame[y:y+h, x:x+w]
            if face.size == 0:
                continue

            # Preprocess face for CNN
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
            gray = gray / 255.0
            gray = np.expand_dims(gray, axis=(0, -1))

            preds = emotion_model.predict(gray, verbose=0)
            emotion_idx = np.argmax(preds)
            emotion = EMOTIONS[emotion_idx]
            confidence = np.max(preds)

            # Draw results
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{emotion} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("Real-Time Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
