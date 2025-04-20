import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
import math

# === Configuration Parameters ===
EAR_THRESHOLD = 0.25
EYE_CLOSED_DURATION_THRESHOLD = 2.0  # seconds

YAWN_THRESHOLD = 30  # pixels
YAWN_TIMEFRAME = 60  # seconds
MAX_YAWNS = 3

HEAD_TILT_ANGLE_THRESHOLD = 35

# === Facial Landmark Indices ===
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
CHIN = 152

# === Mediapipe Setup ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
cap = cv2.VideoCapture(0)

# === Tracking State ===
eye_closed_start = None
yawn_times = deque()

# === Helper Functions ===
def calculate_ear(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C)

def calculate_angle(p1, p2, p3):
    a, b, c = np.array(p1), np.array(p2), np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

# === Main Loop ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)
    current_time = time.time()

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # === Eye Closure Detection ===
            left_eye = np.array([(int(face_landmarks.landmark[i].x * w),
                                  int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE])
            right_eye = np.array([(int(face_landmarks.landmark[i].x * w),
                                   int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE])

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0

            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if avg_ear < EAR_THRESHOLD:
                if eye_closed_start is None:
                    eye_closed_start = current_time
                elif current_time - eye_closed_start >= EYE_CLOSED_DURATION_THRESHOLD:
                    cv2.putText(frame, "ALERT: Eyes Closed Too Long!", (20, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                eye_closed_start = None

            # === Yawning Detection ===
            top_lip = face_landmarks.landmark[13]
            bottom_lip = face_landmarks.landmark[14]
            mouth_gap = abs(int(bottom_lip.y * h) - int(top_lip.y * h))
            cv2.putText(frame, f"Mouth Gap: {mouth_gap}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if mouth_gap > YAWN_THRESHOLD:
                if not yawn_times or current_time - yawn_times[-1] > 1:
                    yawn_times.append(current_time)

            while yawn_times and current_time - yawn_times[0] > YAWN_TIMEFRAME:
                yawn_times.popleft()

            cv2.putText(frame, f"Yawn Count: {len(yawn_times)}", (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            if len(yawn_times) >= MAX_YAWNS:
                cv2.putText(frame, "ALERT: Excessive Yawning!", (20, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Get left and right eye corners
            le = face_landmarks.landmark[LEFT_EYE_CORNER]
            re = face_landmarks.landmark[RIGHT_EYE_CORNER]

            # Convert to (x, y) in pixels
            lx, ly = int(le.x * w), int(le.y * h)
            rx, ry = int(re.x * w), int(re.y * h)

            # Calculate tilt angle (in degrees) of eye line
            dy = ry - ly
            dx = rx - lx
            tilt_angle = np.degrees(np.arctan2(dy, dx))

            cv2.putText(frame, f"Eye Line Tilt: {int(tilt_angle)}°", (20, 210),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)

            # Trigger alert if angle exceeds ±15 degrees
            if abs(tilt_angle) > 15:
                cv2.putText(frame, "ALERT: Head Tilt Detected!", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)


    cv2.imshow("Worker Fatigue Detection", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
