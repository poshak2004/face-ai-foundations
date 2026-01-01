import cv2
import mediapipe as mp

# ----------------------------
# MediaPipe setup
# ----------------------------
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Use model_selection=0 for close faces, 1 for far faces
face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# ----------------------------
# Webcam setup
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

    # OpenCV uses BGR, MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_frame)

    if results.detections:
        h, w, _ = frame.shape

        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box

            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            confidence = detection.score[0]

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + bw, y + bh),
                (0, 255, 0),
                2
            )

            # Confidence label
            cv2.putText(
                frame,
                f"{confidence:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    cv2.imshow("MediaPipe Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ----------------------------
# Cleanup
# ----------------------------
cap.release()
cv2.destroyAllWindows()
