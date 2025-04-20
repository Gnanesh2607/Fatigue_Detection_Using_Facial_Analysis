Fatigue Detection Using Facial Analysis

Project Overview
This project implements a real-time fatigue detection system using facial landmark analysis. It captures webcam video and analyzes signs of drowsiness such as:
- Continuous eye closure
- Frequent yawning
- Abnormal head tilt.

The system uses MediaPipe and OpenCV for landmark detection and video processing.

Technologies Used
Python      - Core programming language
OpenCV      - Video capture and visualization
MediaPipe   - Face mesh and facial landmark detection
NumPy       - Geometric calculations
VS Code     - Code development and execution.

Files Included
fatigue_detector_combined.py - Fully integrated script for real-time detection
eye_closure_detection.py      - Detects continuous eye closure (EAR method)
yawn_detection.py             - Detects yawning using lip distance
head_tilt_detetction.py       - Detects head tilt using face landmark geometry
face_mesh_test.py             - Basic script to test MediaPipe face mesh
requirements.txt              - Lists Python packages required to run the system.

System Features
1. Eye Closure Detection
- Uses Eye Aspect Ratio (EAR)
- Triggers an alert if eyes remain closed ≥ 2 seconds
2. Yawn Detection
- Monitors distance between upper and lower lip
- Counts yawns using a 60-second time window
- Triggers an alert if 3 or more yawns detected
3. Head Tilt Detection
- Measures the tilt angle of the eye line
- Triggers an alert if head is tilted beyond ±15°.

System Workflow
1. Access webcam stream
2. Use MediaPipe to detect facial landmarks
3. Compute EAR, mouth gap, and eye line tilt
4. Compare with thresholds:
   - EAR < 0.25 for ≥2 seconds
   - 3+ yawns in 60s
   - Tilt > ±15°
5. Trigger visual alerts on screen.
   
Future Improvements
- Add audio alerts for higher visibility
- Log alert timestamps for performance reports.
  
Contact – Lingamaneni Gnanesh Chowdary, gcli2607@gmail.com.
