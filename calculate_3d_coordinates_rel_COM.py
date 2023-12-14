import mediapipe as mp
import cv2
import numpy as np
import math
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
# Create a list to store the features (angles)
features = []
total_angles_dict=[]
# Create a Pose object
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Capture video from camera
cap = cv2.VideoCapture(0)
import numpy as np

features_array = []

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def visualize_clustering_real_time(kmeans_model, angles_dict, angle_names, perplexity=5):
    """
    Visualizes the clustering using t-SNE.

    Args:
        kmeans_model (MiniBatchKMeans): The MiniBatchKMeans model.
        angles_dict (dict): Dictionary containing angle annotations as keys and their respective angles as values.
        angle_names (list): List of angle names to use in the KMeans model.
        perplexity (int): Perplexity value for t-SNE.

    Returns:
        None
    """
    global features_array

    # Extract features based on angle_names
    features = np.array([angles_dict[angle_name] for angle_name in angle_names])

    # Append the features to the global features_array
    features_array.append(features)

    # Convert the list of arrays to a single NumPy array
    features_array_np = np.vstack(features_array)

    # Check if the number of samples is sufficient for t-SNE
    if len(features_array) >= perplexity:
        # Use t-SNE for dimensionality reduction
        n_components = min(2, features_array_np.shape[1])
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        reduced_features = tsne.fit_transform(features_array_np)

        # Plot the clustered data
        plt.figure(figsize=(8, 6))

        for i in range(len(angle_names)):
            angle_name = angle_names[i]
            cluster_label = kmeans_model.predict(np.array([angles_dict[angle_name]]))[0]
            plt.scatter(reduced_features[i, 0], reduced_features[i, 1], label=f"{angle_name} (Cluster {cluster_label + 1})")

        plt.title('t-SNE Visualization of Clustering')
        plt.legend()
        plt.show()

    else:
        print(f"Insufficient samples for t-SNE. Current samples: {len(features_array)}, Minimum required: {perplexity}")


def update_kmeans_model(kmeans_model, total_angles_dict, angle_names, min_samples=10, perplexity=5):
    """
    Updates an existing KMeans clustering model with new angles.

    Args:
        kmeans_model (KMeans): The existing KMeans model.
        total_angles_dict (dict): Dictionary containing angle annotations as keys and their respective angles as values.
        angle_names (list): List of angle names to use in the KMeans model.
        min_samples (int): Minimum number of samples required to update the model.
        perplexity (int): Perplexity value for t-SNE.

    Returns:
        None
    """

    # Initialize an empty list to store features
    list_features = []

    # Iterate over each dictionary in total_angles_dict
    for angle_dict in total_angles_dict.values():
        # Initialize an empty list to store features for the current frame
        frame_features = []

        # Iterate over the specified angle_names
        for angle_name in angle_names:
            if angle_name in angle_dict:
                # Add the angle value to the features list
                frame_features.append(angle_dict[angle_name])

        # Add the features for the current frame to the list of features
        list_features.append(frame_features)

    # Convert the list of features to a numpy array
    features_array = np.array(list_features)

    # Check if the number of samples is sufficient for updating
    if len(features_array) >= min_samples:
        # Use t-SNE for dimensionality reduction
        n_components = min(2, features_array.shape[1])
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity)
        reduced_features = tsne.fit_transform(features_array)

        # Update the KMeans model with the new data
        kmeans_model.partial_fit(reduced_features)
    else:
        print(f"Insufficient samples to update the model. Current samples: {len(features_array)}, Minimum required: {min_samples}")

# Example usage:
# Assuming you have a KMeans model initialized, total_angles_dict, and angle_names defined
# kmeans_model = KMeans(n_clusters=3, random_state=42)
# update_kmeans_model(kmeans_model, total_angles_dict, angle_names)
# Assuming the calculate_angle function is defined somewhere in your code

def calculate_angle(a, b, c):
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(math.degrees(radians))
    return angle

def annotate_angles(frame, landmarks):
    """
    Annotates angles on the given frame based on landmarks.

    Args:
        frame: The frame to annotate.
        landmarks: List of pose landmarks.

    Returns:
        dict: Dictionary containing angle annotations as keys and their respective angles as values.
    """
    # Define angle annotations
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
        ("Neck Angle", [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_SHOULDER]),
        ("Spine Angle", [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.NOSE, mp_pose.PoseLandmark.RIGHT_HIP]),
    ]

    # Dictionary to store angle annotations and their respective angles
    angles_dict = {}

    for annotation, landmarks_list in annotations:
        # Calculate the midpoint between the left and right shoulders for the "Neck Angle"
        if mp_pose.PoseLandmark.NOSE in landmarks_list:
            neck_midpoint_x = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x +
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2
            neck_midpoint_y = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                               landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2

            angle = calculate_angle(
                [landmarks[landmark.value].x for landmark in landmarks_list if landmark is not None],
                [landmarks[landmark.value].y for landmark in landmarks_list if landmark is not None],
                [neck_midpoint_x, neck_midpoint_y]
            )
        else:
            angle = calculate_angle(
                [landmarks[landmark.value].x for landmark in landmarks_list],
                [landmarks[landmark.value].y for landmark in landmarks_list],
                [landmarks[landmarks_list[-1].value].x, landmarks[landmarks_list[-1].value].y]
            )

        # Store angles in the dictionary
        angles_dict[annotation] = angle

        text = f"{annotation}: {angle:.2f} degrees"
        cv2.putText(frame, text, (10, 30 + annotations.index((annotation, landmarks_list)) * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return angles_dict
       
def calculate_3d_coordinates(results, frame):
    """
    Calculates 3D coordinates relative to the body's center of mass for all 33 Mediapipe keypoints.

    Args:
        results: The pose detection results from Mediapipe.
        frame: The original frame captured from the camera.

    Returns:
        dict: Dictionary containing landmark names as keys and their respective 3D coordinates as values.
    """
    
    # Mapping landmark numbers to names
    landmark_names = {
        0: "Nose", 1: "Left Eye Inner", 2: "Left Eye", 3: "Left Eye Outer",
        4: "Right Eye Inner", 5: "Right Eye", 6: "Right Eye Outer", 7: "Left Ear",
        8: "Right Ear", 9: "Mouth Left", 10: "Mouth Right", 11: "Left Shoulder",
        12: "Right Shoulder", 13: "Left Elbow", 14: "Right Elbow", 15: "Left Wrist",
        16: "Right Wrist", 17: "Left Pinky", 18: "Right Pinky", 19: "Left Index",
        20: "Right Index", 21: "Left Thumb", 22: "Right Thumb", 23: "Left Hip",
        24: "Right Hip", 25: "Left Knee", 26: "Right Knee", 27: "Left Ankle",
        28: "Right Ankle", 29: "Left Heel", 30: "Right Heel", 31: "Left Foot Index",
        32: "Right Foot Index"
    }

    # Calculate the center of mass coordinates
    com_x = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x +
             results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2
    com_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y +
             results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
    com_z = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].z +
             results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].z) / 2

    # Dictionary to store landmark names and their respective 3D coordinates
    coordinates_dict = {}

    # Iterate through each pose landmark
    for i in range(33):
        # Extract the landmark object
        landmark = results.pose_landmarks.landmark[i]

        # Calculate normalized 3D coordinates relative to the center of mass
        x_rel = (landmark.x - com_x) * frame.shape[1]
        y_rel = (landmark.y - com_y) * frame.shape[0]
        z_rel = (landmark.z - com_z) * (frame.shape[1] + frame.shape[0]) / 2

        # Store coordinates in the dictionary
        landmark_name = landmark_names[i]
        coordinates_dict[landmark_name] = (x_rel, y_rel, z_rel)

        # Display landmark name and relative 3D coordinates on the right side of the frame in green
        text = f"{landmark_name}: ({x_rel:.2f}, {y_rel:.2f}, {z_rel:.2f})"
        text_position = (frame.shape[1] - 250, 30 + 15 * i)  # Adjusted the spacing
        cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    return coordinates_dict



num_clusters = 3  # You can adjust this number
kmeans_model = MiniBatchKMeans(n_clusters=num_clusters, random_state=42,n_init="auto")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform pose detection
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Extract landmarks
        landmarks = results.pose_landmarks.landmark

        # Calculate 3D coordinates relative to the center of mass for all landmarks
        calculate_3d_coordinates(results, frame)

        # Annotate angles on the image
        angles_dict=annotate_angles(frame, landmarks)
        total_angles_dict+=angles_dict

        # Draw annotations on the image
        #draw_annotations(frame, angles_dict)

        # Update the KMeans model with the new angles
        update_kmeans_model(kmeans_model, angles_dict, ["Right Elbow Angle","Right Wrist Angle"])

        # Visualize the clustering in real-time
        visualize_clustering_real_time(kmeans_model, angles_dict, ["Right Elbow Angle","Right Wrist Angle"], perplexity=5)


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
