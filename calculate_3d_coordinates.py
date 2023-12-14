import mediapipe as mp
import cv2

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Create a Pose object
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from camera
cap = cv2.VideoCapture(0)

def calculate_3d_coordinates(results, frame):
    """
    Calculates 3D coordinates for all 33 Mediapipe keypoints.

    Args:
        results: The pose detection results from Mediapipe.
        frame: The original frame captured from the camera.

    Returns:
        None
    """


    # Iterate through each pose landmark
    for i in range(33):
        # Extract the landmark object
        landmark = results.pose_landmarks.landmark[i]

        # Calculate normalized 3D coordinates
        x_3d = landmark.x * frame.shape[1]
        y_3d = landmark.y * frame.shape[0]
        z_3d = landmark.z * (frame.shape[1] + frame.shape[0]) / 2

        # Print the landmark ID and 3D coordinates
        print(f"Landmark ID: {i}, 3D coordinates: ({x_3d:.2f}, {y_3d:.2f}, {z_3d:.2f})")


while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Calculate 3D coordinates for all landmarks
        calculate_3d_coordinates(results, frame)

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
