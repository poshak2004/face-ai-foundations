import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ----------------------------
# Load model
# ----------------------------
MODEL_PATH = "models/blaze_face_short_range.tflite"

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE
)

detector = vision.FaceDetector.create_from_options(options)

# ----------------------------
# Webcam
# ----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not accessible")

print("Webcam started. Press 'q' to quit.")

# ----------------------------
# Main loop
# ----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    detection_result = detector.detect(mp_image)

    if detection_result.detections:
        h, w, _ = frame.shape

        for detection in detection_result.detections:
            bbox = detection.bounding_box
            confidence = detection.categories[0].score

            x = int(bbox.origin_x)
            y = int(bbox.origin_y)
            bw = int(bbox.width)
            bh = int(bbox.height)

            cv2.rectangle(
                frame,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("MediaPipe Face Detection (Tasks API)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
