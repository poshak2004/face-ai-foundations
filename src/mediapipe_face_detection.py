import cv2
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MODEL_PATH = "models/blaze_face_short_range.tflite"
SAVE_DIR = "datasets/faces/raw"
CONFIDENCE_THRESHOLD = 0.6

os.makedirs(SAVE_DIR, exist_ok=True)

# --------------------------------------------------
# MediaPipe Face Detector (Tasks API)
# --------------------------------------------------
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.FaceDetector.create_from_options(options)

# --------------------------------------------------
# Webcam
# --------------------------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("❌ Webcam not accessible")

print("✅ Webcam started")
print("📸 Collecting face images... Press 'q' to stop")

face_count = 0

# --------------------------------------------------
# Main Loop
# --------------------------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB (MediaPipe requirement)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect(mp_image)

    if detection_result.detections:
        h, w, _ = frame.shape

        for detection in detection_result.detections:
            score = detection.categories[0].score
            if score < CONFIDENCE_THRESHOLD:
                continue

            bbox = detection.bounding_box

            x = max(0, bbox.origin_x)
            y = max(0, bbox.origin_y)
            bw = bbox.width
            bh = bbox.height

            # Crop face
            face = frame[y:y+bh, x:x+bw]

            if face.size == 0:
                continue

            face_count += 1
            face_path = f"{SAVE_DIR}/face_{face_count}.jpg"
            cv2.imwrite(face_path, face)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{score:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    cv2.imshow("MediaPipe Face Detection + Face Cropping", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# --------------------------------------------------
# Cleanup
# --------------------------------------------------
cap.release()
cv2.destroyAllWindows()

print(f"✅ Done. Total faces saved: {face_count}")
print(f"📁 Saved in: {SAVE_DIR}")
