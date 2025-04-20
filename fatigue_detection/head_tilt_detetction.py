import cv2
import mediapipe as mp
import numpy as np
import math

# Mediapipe Setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Webcam
cap = cv2.VideoCapture(0)

# Landmark indexes
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
CHIN = 152

def calculate_angle(p1, p2, p3):
    # Convert 3 points to vectors and find angle between vectors
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmark coordinates
            left_eye = face_landmarks.landmark[LEFT_EYE_CORNER]
            right_eye = face_landmarks.landmark[RIGHT_EYE_CORNER]
            chin = face_landmarks.landmark[CHIN]

            # Convert to pixel coordinates
            p1 = (int(left_eye.x * w), int(left_eye.y * h))
            p2 = (int(chin.x * w), int(chin.y * h))
            p3 = (int(right_eye.x * w), int(right_eye.y * h))

            # Calculate angle between eyes and chin
            angle = calculate_angle(p1, p2, p3)

            # Display angle
            cv2.putText(frame, f"Head Tilt Angle: {int(angle)}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if angle < 120:  # If angle too narrow, head is likely tilted
                cv2.putText(frame, "FATIGUE ALERT: Head Tilt Detected", (30, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("Fatigue Detection - Head Tilt", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
