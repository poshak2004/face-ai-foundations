import cv2
import numpy as np
import tensorflow as tf
from collections import deque
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------------------
# CONFIG
# ----------------------------
MODEL_PATH = "models/gender_mobilenet.keras"
FACE_MODEL = "models/blaze_face_short_range.tflite"
IMG_SIZE = 224

LABELS = ["Female", "Male"]

# ----------------------------
# LOAD MODEL
# ----------------------------
gender_model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Gender model loaded")

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

print("🎥 Real-time Gender Detection started (press 'q' to quit)")

# ----------------------------
# TEMPORAL SMOOTHING
# ----------------------------
history = deque(maxlen=7)

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

            # 🔥 TIGHTER CROP
            pad = int(0.1 * min(w, h))
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(frame.shape[1], x + w + pad)
            y2 = min(frame.shape[0], y + h + pad)

            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            # ----------------------------
            # PREPROCESS (CORRECT)
            # ----------------------------
            face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
            face = preprocess_input(face.astype("float32"))
            face = np.expand_dims(face, axis=0)

            pred = gender_model.predict(face, verbose=0)[0][0]
            history.append(pred)

            avg_pred = sum(history) / len(history)
            label = LABELS[int(avg_pred > 0.5)]
            confidence = avg_pred if label == "Male" else 1 - avg_pred

            # ----------------------------
            # DRAW
            # ----------------------------
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("Real-Time Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
