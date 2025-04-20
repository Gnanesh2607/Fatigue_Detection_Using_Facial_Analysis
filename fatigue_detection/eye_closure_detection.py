import cv2
import mediapipe as mp
import numpy as np
import time

def calculate_ear(eye_landmarks):
    # Eye landmarks: [left, top-left, top-right, right, bottom-right, bottom-left]
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])  # vertical
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])  # vertical
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])  # horizontal
    ear = (A + B) / (2.0 * C)
    return ear

# EAR threshold for closed eye
EAR_THRESHOLD = 0.25
CLOSED_DURATION_THRESHOLD = 2.0  # seconds

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# Eye landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Webcam
cap = cv2.VideoCapture(0)
eye_closed_start = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    h, w, _ = frame.shape

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get coordinates for both eyes
            left_eye = np.array([(int(face_landmarks.landmark[i].x * w),
                                  int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE])
            right_eye = np.array([(int(face_landmarks.landmark[i].x * w),
                                   int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE])

            # Calculate EAR for both eyes
            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            # Display EAR value
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Check if eyes are closed
            if avg_ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                else:
                    closed_duration = time.time() - eye_closed_start
                    if closed_duration > CLOSED_DURATION_THRESHOLD:
                        cv2.putText(frame, "FATIGUE ALERT: Eyes closed!", (30, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                eye_closed_start = None

    cv2.imshow("Fatigue Detection - Eye Closure", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
