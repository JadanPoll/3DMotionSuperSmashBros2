import mediapipe as mp
import cv2
import numpy as np


def rotate_coordinates_towards_camera(landmarks, camera_direction=(0, 0, 1)):
    """
    Rotates all 3D coordinates in `landmarks` towards the camera's forward direction.

    Args:
        landmarks: An array of 3D coordinates for all 33 Mediapipe landmarks (Nx3).
        camera_direction: A 3D vector representing the camera's forward direction.

    Returns:
        rotated_landmarks: An array of 3D coordinates after rotation (Nx3).
    """

    # Extract nose landmark
    nose_landmark = landmarks[mp_pose.PoseLandmark.NOSE]

    # Calculate rotation angle
    rotation_angle = np.arctan2(
        nose_landmark[1] - camera_direction[1],
        nose_landmark[0] - camera_direction[0],
    )

    # Create rotation matrix
    rotation_matrix = cv2.Rodrigues(np.array([0, rotation_angle, 0]))[0]

    # Rotate all landmarks
    rotated_landmarks = (rotation_matrix @ landmarks.T).T

    return rotated_landmarks


# Capture and process frames
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)


cap = cv2.VideoCapture(0)


# Set capture resolution to screen resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(frame_rgb)

    # Check if any pose landmarks are detected
    if results.pose_landmarks:
        # Extract and rotate landmark coordinates
        landmarks = np.array([[l.x, l.y, l.z] for l in results.pose_landmarks.landmark])
        rotated_landmarks = rotate_coordinates_towards_camera(landmarks)

        # Update the landmark object with rotated coordinates
        for i, landmark in enumerate(rotated_landmarks):
            results.pose_landmarks.landmark[i].x, results.pose_landmarks.landmark[i].y, results.pose_landmarks.landmark[i].z = landmark

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
