
import cv2
import mediapipe as mp
import numpy as np
from math import degrees, atan2

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Draw circles and angles with error handling

class MyPoseLandmarkIndices:
    NOSE = mp.solutions.pose.PoseLandmark.NOSE
    LEFT_EYE_INNER = mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER
    LEFT_EYE = mp.solutions.pose.PoseLandmark.LEFT_EYE
    LEFT_EYE_OUTER = mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER
    RIGHT_EYE_INNER = mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER
    RIGHT_EYE = mp.solutions.pose.PoseLandmark.RIGHT_EYE
    RIGHT_EYE_OUTER = mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER
    LEFT_EAR = mp.solutions.pose.PoseLandmark.LEFT_EAR
    RIGHT_EAR = mp.solutions.pose.PoseLandmark.RIGHT_EAR
    MOUTH_LEFT = mp.solutions.pose.PoseLandmark.MOUTH_LEFT
    MOUTH_RIGHT = mp.solutions.pose.PoseLandmark.MOUTH_RIGHT
    LEFT_SHOULDER = mp.solutions.pose.PoseLandmark.LEFT_SHOULDER
    RIGHT_SHOULDER = mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER
    LEFT_ELBOW = mp.solutions.pose.PoseLandmark.LEFT_ELBOW
    RIGHT_ELBOW = mp.solutions.pose.PoseLandmark.RIGHT_ELBOW
    LEFT_WRIST = mp.solutions.pose.PoseLandmark.LEFT_WRIST
    RIGHT_WRIST = mp.solutions.pose.PoseLandmark.RIGHT_WRIST
    LEFT_PINKY = mp.solutions.pose.PoseLandmark.LEFT_PINKY
    RIGHT_PINKY = mp.solutions.pose.PoseLandmark.RIGHT_PINKY
    LEFT_INDEX = mp.solutions.pose.PoseLandmark.LEFT_INDEX
    RIGHT_INDEX = mp.solutions.pose.PoseLandmark.RIGHT_INDEX
    LEFT_THUMB = mp.solutions.pose.PoseLandmark.LEFT_THUMB
    RIGHT_THUMB = mp.solutions.pose.PoseLandmark.RIGHT_THUMB
    LEFT_HIP = mp.solutions.pose.PoseLandmark.LEFT_HIP
    RIGHT_HIP = mp.solutions.pose.PoseLandmark.RIGHT_HIP
    LEFT_KNEE = mp.solutions.pose.PoseLandmark.LEFT_KNEE
    RIGHT_KNEE = mp.solutions.pose.PoseLandmark.RIGHT_KNEE
    LEFT_ANKLE = mp.solutions.pose.PoseLandmark.LEFT_ANKLE
    RIGHT_ANKLE = mp.solutions.pose.PoseLandmark.RIGHT_ANKLE
    LEFT_HEEL = mp.solutions.pose.PoseLandmark.LEFT_HEEL
    RIGHT_HEEL = mp.solutions.pose.PoseLandmark.RIGHT_HEEL
    LEFT_FOOT_INDEX = mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX
    RIGHT_FOOT_INDEX = mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX
    # Draw smaller arcs with improved aesthetics
def draw_circles_and_angles(image, landmarks, joint_indices, color, side, ellipse_scale=0.5, straight_line_threshold=160):
    for joint_set in joint_indices:
        try:
            joint_positions = [landmarks[joint_index] for joint_index in joint_set]
            if None not in joint_positions:  # Check if all required landmarks are present
                joint_positions = np.array([(int(joint.x * image.shape[1]), int(joint.y * image.shape[0])) for joint in joint_positions])

                # Calculate angle
                angle = calculate_angle(*joint_positions)

                # Draw straight line if angle exceeds the threshold
                if angle > straight_line_threshold:
                    cv2.line(image, tuple(joint_positions[0]), tuple(joint_positions[2]), color, 2)

                else:
                    # Draw smaller arc along the line segments connecting joints
                    p1, p2, p3 = joint_positions
                    angle1 = degrees(atan2(p1[1] - p2[1], p1[0] - p2[0]))
                    angle2 = degrees(atan2(p3[1] - p2[1], p3[0] - p2[0]))

                    # Determine the orientation of the angle
                    if angle1 < angle2:
                        angle1, angle2 = angle2, angle1  # Swap angles for acute angles

                    # Calculate midpoint of the ellipse
                    center = tuple(joint_positions[1])

                    # Calculate minor and major axes lengths
                    minor_axis = int(np.linalg.norm(p1 - p2) * ellipse_scale)
                    major_axis = int(np.linalg.norm(p3 - p2) * ellipse_scale)

                    # Draw the arc
                    cv2.ellipse(image, center, (major_axis, minor_axis), 0, angle1, angle2, color, 2)

                    # Determine text position (midpoint of the ellipse)
                    text_position = (center[0] + major_axis // 2 - 15, center[1] + minor_axis // 2 + 15)

                    # Draw angle measure
                    cv2.putText(image, f'{angle:.2f}', text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)

        except Exception as e:
            print(f"Error processing {side} side: {e}")

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


# Function for updating and displaying clusters
def update_and_display_clusters(data_array, kmeans):
    labels = kmeans.predict(data_array)
    plt.scatter(data_array[:, 0], data_array[:, 1], c=labels, cmap='viridis')
    plt.pause(0.0001)  # Add a small pause to allow the plot to update (non-blocking)



def display_landmark(image, text, value, position, font_scale=1, color=(0, 0, 255), thickness=2):
    """
    Display a landmark value on the image.

    Parameters:
    - image: The image to display the text on.
    - text: The label for the landmark.
    - value: The value of the landmark.
    - position: The position to display the text on the image.
    - font_scale: Font scale for the text (default is 1).
    - color: Color of the text (default is white).
    - thickness: Thickness of the text (default is 2).
    """
    if value is not None:
        if isinstance(value, np.ndarray):
            rounded_value = [round(element, 2) for element in value]
            text_to_display = f"{text}: {rounded_value}"
        else:
            # If it's not a NumPy array, round the value directly
            rounded_value = round(value, 2)
            text_to_display = f"{text}: {rounded_value}"

    else:
        text_to_display = f"{text} is not available"

    cv2.putText(image, text_to_display, position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

mp_drawing = mp.solutions.drawing_utils
global mp_pose
mp_pose = mp.solutions.pose

# Initialize pose estimation model
pose = mp_pose.Pose()

# Capture video stream
cap = cv2.VideoCapture(0)

# Set window properties to full screen
cv2.namedWindow('MediaPipe Pose Estimation', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('MediaPipe Pose Estimation', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)



# Initialize KMeans with the desired number of clusters
kmeans = KMeans(n_clusters=3, random_state=42,n_init='auto')
data=[]
kmeansi=0
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
    left_hand_position = None
    left_elbow_angle = None
    left_shoulder_rotation = None
    left_hip_rotation = None
    left_body_lean = None
    left_hand_wrist_orientation = None

    left_knee_angle = None
    
    right_hand_position = None
    right_elbow_angle = None
    right_shoulder_rotation = None
    right_hip_rotation = None
    right_body_lean = None
    right_hand_wrist_orientation = None

    right_knee_angle = None
    # Calculate features if pose landmarks are available
    if pose_landmarks:

        # Left side features
        left_side_joint_sets = [\
            (MyPoseLandmarkIndices.LEFT_WRIST, MyPoseLandmarkIndices.LEFT_ELBOW, MyPoseLandmarkIndices.LEFT_SHOULDER),\
            (MyPoseLandmarkIndices.LEFT_SHOULDER, MyPoseLandmarkIndices.LEFT_HIP, MyPoseLandmarkIndices.LEFT_ELBOW),\
            (MyPoseLandmarkIndices.LEFT_HIP, MyPoseLandmarkIndices.LEFT_KNEE, MyPoseLandmarkIndices.LEFT_ANKLE)\
        ]

        draw_circles_and_angles(image, pose_landmarks, left_side_joint_sets, (0, 255, 0), 'Left')

        # Right side features
        right_side_joint_sets = [\
            (MyPoseLandmarkIndices.RIGHT_WRIST, MyPoseLandmarkIndices.RIGHT_ELBOW, MyPoseLandmarkIndices.RIGHT_SHOULDER),\
            (MyPoseLandmarkIndices.RIGHT_SHOULDER, MyPoseLandmarkIndices.RIGHT_HIP, MyPoseLandmarkIndices.RIGHT_ELBOW),\
            (MyPoseLandmarkIndices.RIGHT_HIP, MyPoseLandmarkIndices.RIGHT_KNEE, MyPoseLandmarkIndices.RIGHT_ANKLE)\
        ]

        draw_circles_and_angles(image, pose_landmarks, right_side_joint_sets, (0, 255, 0), 'Right')



        # Left side
        left_shoulder = np.array([pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                  pose_landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y])
        left_elbow = np.array([pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                               pose_landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y])

        left_hand = np.array([pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                             pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])

        left_wrist = np.array([pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              pose_landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
        # Left knee, hip, and ankle landmarks
        
        left_hip = np.array([pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                             pose_landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y])

        left_knee = np.array([pose_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                              pose_landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y])

        left_ankle = np.array([pose_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                               pose_landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y])


        # Right side
        right_shoulder = np.array([pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                   pose_landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y])
        right_elbow = np.array([pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                pose_landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y])

        right_hand = np.array([pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                              pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
        
        right_wrist = np.array([pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
        
            # Right knee, hip, and ankle landmarks
        right_knee = np.array([pose_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                               pose_landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y])
        right_hip = np.array([pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                              pose_landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y])
        right_ankle = np.array([pose_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                pose_landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y])


# Calculate features if pose landmarks are available

        # Left side features
        left_hand_position = left_hand
        left_elbow_angle = calculate_angle(left_wrist, left_elbow, left_shoulder)
        left_shoulder_rotation = calculate_angle(left_shoulder, left_elbow, left_wrist)
        left_hip_rotation = calculate_angle(left_hip, left_shoulder, left_elbow)
        left_body_lean = left_shoulder[0] - left_hip[0]
        left_hand_wrist_orientation = calculate_angle(left_hand, left_wrist, left_elbow)

        # Calculate angles
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)


        # Right side features
        right_hand_position = right_hand
        right_elbow_angle = calculate_angle(right_wrist, right_elbow, right_shoulder)
        right_shoulder_rotation = calculate_angle(right_shoulder, right_elbow, right_wrist)
        right_hip_rotation = calculate_angle(right_hip, right_shoulder, right_elbow)
        right_body_lean = right_shoulder[0] - right_hip[0]
        right_hand_wrist_orientation = calculate_angle(right_hand, right_wrist, right_elbow)

        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)



        '''
        # Display left side features
        display_landmark(image, "Left Hand Position", left_hand_position, (10, 30))
        display_landmark(image, "Left Elbow Angle", left_elbow_angle, (10, 60))
        display_landmark(image, "Left Shoulder Rotation", left_shoulder_rotation, (10, 90))
        display_landmark(image, "Left Hip Rotation", left_hip_rotation, (10, 120))
        display_landmark(image, "Left Body Lean", left_body_lean, (10, 150))
        display_landmark(image, "Left Hand-Wrist Orientation", left_hand_wrist_orientation, (10, 180))


        display_landmark(image, "Left Knee Angle", left_knee_angle, (10, 210))

        # Display right side features
        display_landmark(image, "Right Hand Position", right_hand_position, (10, 240))
        display_landmark(image, "Right Elbow Angle", right_elbow_angle, (10, 270))
        display_landmark(image, "Right Shoulder Rotation", right_shoulder_rotation, (10, 300))
        display_landmark(image, "Right Hip Rotation", right_hip_rotation, (10, 330))
        display_landmark(image, "Right Body Lean", right_body_lean, (10, 360))
        display_landmark(image, "Right Hand-Wrist Orientation", right_hand_wrist_orientation, (10, 390))
        display_landmark(image, "Right Knee Angle", right_knee_angle, (10, 420))
        '''
        # Update the dataset with the new data point
        new_data_point = [left_hand_position[0], left_hand_position[1]]
        data.append(new_data_point)

        if kmeansi<3:
            kmeansi+=1
        else:

            # Convert the dataset to a NumPy array
            data_array = np.array(data)

            # Update the KMeans model with the new data
            kmeans.fit(data_array)


        # Call the function for updating and displaying clusters
            update_and_display_clusters(data_array, kmeans)

    # Draw landmarks on the image
    if results.pose_landmarks is not None:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Show the output image
    cv2.imshow('MediaPipe Pose Estimation', image)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release capture object and destroy all windows
cap.release()
cv2.destroyAllWindows()
