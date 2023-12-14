import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create a Pose object
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from camera
cap = cv2.VideoCapture(0)

# Initialize Kalman filters for each landmark
kalman_filters = [cv2.KalmanFilter(4, 2) for _ in range(33)]

# Define outlier detection threshold
outlier_threshold = 3
def calculate_3d_coordinates(results, frame):
    smoothed_landmarks = []

    # Iterate through each pose landmark
    for i in range(33):
        landmark = results.pose_landmarks.landmark[i]

        # Calculate normalized 3D coordinates
        x_3d = int(landmark.x * frame.shape[1])
        y_3d = int(landmark.y * frame.shape[0])
        z_3d = int(landmark.z * (frame.shape[1] + frame.shape[0]) / 2)

        # Apply outlier detection
        if (
            abs(x_3d - kalman_filters[i].statePost[0][0]) > outlier_threshold
            or abs(y_3d - kalman_filters[i].statePost[1][0]) > outlier_threshold
            or abs(z_3d - kalman_filters[i].statePost[2][0]) > outlier_threshold
        ):
            kalman_filters[i].statePost = np.array([x_3d, y_3d, z_3d, 0, 0, 0], dtype=np.float32)

        # Apply Kalman filter
        kalman_filters[i].predict()
        measurement = np.array([x_3d, y_3d, z_3d])
        kalman_filters[i].correct(measurement)

        # Update smoothed landmarks
        smoothed_landmarks.append([kalman_filters[i].statePost[0][0],
                                   kalman_filters[i].statePost[1][0],
                                   kalman_filters[i].statePost[2][0]])

    return smoothed_landmarks

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
            # Calculate smoothed 3D coordinates
            smoothed_landmarks = calculate_3d_coordinates(results, frame)

            # Draw the pose landmarks on the frame
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Draw the smoothed landmarks
            for landmark in smoothed_landmarks:
                x, y, _ = landmark
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

    except Exception as e:
        # Handle exceptions gracefully
        print(f"Error during pose processing: {e}")
        continue

    # Display the resulting frame
    cv2.imshow('Mediapipe Pose', frame)

    # Quit when Q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
