import mediapipe as mp
import cv2

cap = cv2.VideoCapture(0)

facemesh = mp.solutions.face_mesh
face_mesh = facemesh.FaceMesh(static_image_mode=True, min_tracking_confidence=0.6, min_detection_confidence=0.6)
draw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame from camera. Exiting...")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_frame)
    if result.multi_face_landmarks:
        for face_landmarks in result.multi_face_landmarks:
            draw.draw_landmarks(frame, face_landmarks)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()