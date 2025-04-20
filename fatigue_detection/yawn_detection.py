import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

# Initialize Mediapipe
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

# Webcam
cap = cv2.VideoCapture(0)

# Yawn detection config
YAWN_THRESHOLD = 30  # mouth opening (in pixels)
YAWN_TIMEFRAME = 60  # seconds
MAX_YAWNS = 3

yawn_times = deque()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get upper and lower lip points
            top_lip = face_landmarks.landmark[13]
            bottom_lip = face_landmarks.landmark[14]

            # Convert to pixel coordinates
            top_y = int(top_lip.y * h)
            bottom_y = int(bottom_lip.y * h)

            mouth_gap = abs(bottom_y - top_y)

            # Show mouth gap
            cv2.putText(frame, f"Mouth Gap: {mouth_gap}", (30, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

