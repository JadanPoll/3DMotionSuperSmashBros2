import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create a Pose object
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from camera
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(math.degrees(radians))
    return angle

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    try:
        # Perform pose detection
        results = pose.process(frame_rgb)

        # Check if landmarks are present and have sufficient confidence
        if results.pose_landmarks and all(
            lm.visibility > 0.5 for lm in results.pose_landmarks.landmark
        ):
            # Extract landmarks
            landmarks = results.pose_landmarks.landmark

            # Annotate each angle on the image
            annotations = [
                ("Shoulder Angle", [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW]),
                ("Left Elbow Angle", [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]),
                ("Right Elbow Angle", [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]),
                ("Left Wrist Angle", [mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST, mp_pose.PoseLandmark.LEFT_THUMB]),
                ("Right Wrist Angle", [mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST, mp_pose.PoseLandmark.RIGHT_THUMB]),
                ("Hip Angle", [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.LEFT_KNEE]),
                ("Left Knee Angle", [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE]),
                ("Right Knee Angle", [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]),
                ("Left Ankle Angle", [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.LEFT_HEEL]),
                ("Right Ankle Angle", [mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE, mp_pose.PoseLandmark.RIGHT_HEEL]),
                ("Neck Angle", [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.NECK, mp_pose.PoseLandmark.RIGHT_SHOULDER]),
                ("Spine Angle", [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.SPINE, mp_pose.PoseLandmark.RIGHT_HIP]),
            ]

            for annotation, landmarks_list in annotations:
                angle = calculate_angle(
                    [landmarks[landmark.value].x for landmark in landmarks_list],
                    [landmarks[landmark.value].y for landmark in landmarks_list],
                    [landmarks[landmarks_list[-1].value].x, landmarks[landmarks_list[-1].value].y]
                )
                text = f"{annotation}: {angle:.2f} degrees"
                cv2.putText(frame, text, (10, 30 + annotations.index((annotation, landmarks_list)) * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error during pose processing: {e}")
        continue

    # Draw the pose landmarks on the frame
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow('Mediapipe Pose', frame)

    # Quit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
