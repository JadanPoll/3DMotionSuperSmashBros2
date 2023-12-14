import cv2
import mediapipe as mp
import numpy as np
from math import degrees, atan2

class KickType:
    FRONT_KICK = "Front Kick"
    ROUNDHOUSE_KICK = "Roundhouse Kick"
    SIDE_KICK = "Side Kick"
    NO_KICK = "No Kick"

class PunchType:
    JAB = "Jab"
    CROSS = "Cross"
    HOOK = "Hook"
    NO_PUNCH = "No Punch"

# Default threshold values
front_kick_threshold = 0.2
roundhouse_kick_threshold = 0.2
side_kick_threshold = 0.2

jab_threshold = 0.15
cross_threshold = 0.15
hook_threshold = 0.15

def detect_kick_type(knee, ankle, hip):

    # Calculate angles
    angle_knee_ankle = calculate_angle(hip, knee, ankle)
    angle_hip_knee = calculate_angle(ankle, hip, knee)

    # Determine kick type based on angles and positions
    if angle_knee_ankle > front_kick_threshold:
        # If the knee-ankle angle is greater than the front kick threshold, classify as Front Kick
        return KickType.FRONT_KICK, angle_knee_ankle, angle_hip_knee
    elif angle_hip_knee > roundhouse_kick_threshold:
        # If the hip-knee angle is greater than the roundhouse kick threshold, classify as Roundhouse Kick
        return KickType.ROUNDHOUSE_KICK, angle_knee_ankle, angle_hip_knee
    elif angle_knee_ankle < side_kick_threshold:
        # If the knee-ankle angle is less than the side kick threshold, classify as Side Kick
        return KickType.SIDE_KICK, angle_knee_ankle, angle_hip_knee
    else:
        # If none of the conditions are met, classify as No Kick
        return KickType.NO_KICK, angle_knee_ankle, angle_hip_knee

def detect_punch_type(hand, elbow, shoulder):

    # Calculate angles
    angle_hand_elbow = calculate_angle(shoulder, hand, elbow)
    angle_elbow_shoulder = calculate_angle(hand, elbow, shoulder)

    # Determine punch type based on angles and positions
    if angle_hand_elbow < jab_threshold:
        # If the hand-elbow angle is less than the jab threshold, classify as Jab
        return PunchType.JAB, angle_hand_elbow, angle_elbow_shoulder
    elif angle_elbow_shoulder > cross_threshold:
        # If the elbow-shoulder angle is greater than the cross threshold, classify as Cross
        return PunchType.CROSS, angle_hand_elbow, angle_elbow_shoulder
    elif angle_hand_elbow > hook_threshold:
        # If the hand-elbow angle is greater than the hook threshold, classify as Hook
        return PunchType.HOOK, angle_hand_elbow, angle_elbow_shoulder
    else:
        # If none of the conditions are met, classify as No Punch
        return PunchType.NO_PUNCH, angle_hand_elbow, angle_elbow_shoulder

# Function to calculate angle between two vectors (points)
def calculate_angle(point1, center_point, point2):
    vector1 = np.array([point1[0] - center_point[0], point1[1] - center_point[1]])
    vector2 = np.array([point2[0] - center_point[0], point2[1] - center_point[1]])

    # Check if either vector has zero magnitude
    if np.linalg.norm(vector1) == 0 or np.linalg.norm(vector2) == 0:
        return 0.0

    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))

    # Handle edge cases where cos_theta is slightly greater than 1 or less than -1 due to numerical precision
    cos_theta = min(max(cos_theta, -1.0), 1.0)

    # Calculate and return angle in degrees
    return degrees(np.arccos(cos_theta))

# Function to update thresholds from sliders
def update_thresholds(x):
    global front_kick_threshold, roundhouse_kick_threshold, side_kick_threshold
    global jab_threshold, cross_threshold, hook_threshold

    front_kick_threshold = cv2.getTrackbarPos('Front Kick', 'Thresholds') / 100.0
    roundhouse_kick_threshold = cv2.getTrackbarPos('Roundhouse Kick', 'Thresholds') / 100.0
    side_kick_threshold = cv2.getTrackbarPos('Side Kick', 'Thresholds') / 100.0

    jab_threshold = cv2.getTrackbarPos('Jab', 'Thresholds') / 100.0
    cross_threshold = cv2.getTrackbarPos('Cross', 'Thresholds') / 100.0
    hook_threshold = cv2.getTrackbarPos('Hook', 'Thresholds') / 100.0

# Initialize pose estimation model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Capture video stream
cap = cv2.VideoCapture(0)

# Create a window for sliders
cv2.namedWindow('Thresholds')


# Create trackbars for each threshold
cv2.createTrackbar('Front Kick', 'Thresholds', int(front_kick_threshold * 100), 100, update_thresholds)
cv2.createTrackbar('Roundhouse Kick', 'Thresholds', int(roundhouse_kick_threshold * 100), 100, update_thresholds)
cv2.createTrackbar('Side Kick', 'Thresholds', int(side_kick_threshold * 100), 100, update_thresholds)

cv2.createTrackbar('Jab', 'Thresholds', int(jab_threshold * 100), 100, update_thresholds)
cv2.createTrackbar('Cross', 'Thresholds', int(cross_threshold * 100), 100, update_thresholds)
cv2.createTrackbar('Hook', 'Thresholds', int(hook_threshold * 100), 100, update_thresholds)
while True:
    # Capture frame-by-frame
    success, image = cap.read()

    # Convert image to RGB format for MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform pose estimation
    results = pose.process(image_rgb)

    # Extract pose landmarks
    pose_landmarks = results.pose_landmarks.landmark if results.pose_landmarks else []

    # Initialize variables for features
    if pose_landmarks:
        right_knee = np.array([pose_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                               pose_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
        right_ankle = np.array([pose_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                pose_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])
        right_hip = np.array([pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])

        left_hand = np.array([pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
        left_elbow = np.array([pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])
        left_shoulder = np.array([pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])

        # Detect kick and punch types
        kick_type, angle_knee_ankle, angle_hip_knee = detect_kick_type(right_knee, right_ankle, right_hip)
        punch_type, angle_hand_elbow, angle_elbow_shoulder = detect_punch_type(left_hand, left_elbow, left_shoulder)

        # Draw text indicating the detected kick and punch types and angles
        cv2.putText(image, f"Kick Type: {kick_type}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                    cv2.LINE_AA)
        cv2.putText(image, f"Punch Type: {punch_type}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                    cv2.LINE_AA)
        cv2.putText(image, f"Knee-Ankle Angle: {angle_knee_ankle:.2f} degrees", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Hip-Knee Angle: {angle_hip_knee:.2f} degrees", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Hand-Elbow Angle: {angle_hand_elbow:.2f} degrees", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(image, f"Elbow-Shoulder Angle: {angle_elbow_shoulder:.2f} degrees", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    # Draw landmarks on the image
    if results.pose_landmarks is not None:
        mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the output image
    cv2.imshow('Action Detection', image)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
