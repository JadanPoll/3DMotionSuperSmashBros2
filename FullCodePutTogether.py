# Import necessary libraries
import cv2
import mediapipe as mp
from sklearn.cluster import DBSCAN
from hmmlearn import hmm
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import StandardScaler
import time
import pyautogui
import pygetwindow as gw
from pose_analysis import *
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

# Function for Hidden Markov Model
def train_hmm_model(sequences, num_hidden_states):
    model = hmm.GaussianHMM(n_components=num_hidden_states, covariance_type="full", n_iter=1000)
    model.fit(sequences)
    return model

# Function to identify transitions and classify fighting moves
def identify_transitions_using_hmm(features, clusters, hmm_model):
    # Implement logic to detect transitions using HMM
    # Update clustered poses based on transitions

    # Ensure that features is a NumPy array or convert DataFrame to NumPy array
    if isinstance(features, pd.DataFrame):
        features = features.values
    
    return classified_moves

# Constants
MIN_SAMPLES_FOR_DBSCAN = 50
MAX_SAMPLES_FOR_DBSCAN = 500
EPSILON_FOR_DBSCAN = 0.51
SELECTED_COLUMNS_FOR_DBSCAN = ["Nose"]
pca = PCA(n_components=2)
scaler = StandardScaler()
def main():
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Initialize DataFrames
    df_coords_features = pd.DataFrame()
    df_angles_features = pd.DataFrame()

    fig, ax = plt.subplots(figsize=(8, 6))

    cap = cv2.VideoCapture(0)  # Use appropriate camera index if not the default camera

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(frame_rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            rel_coords_features = calculate_relative_3d_coordinates(mp_pose,results, frame, x_quantization=1.0,
                                                           y_quantization=1.0, z_quantization=1.0)

            df_coords_features = update_dataframe(df_coords_features, rel_coords_features)

            angles_features = calculate_pose_angles(frame, landmarks,mp_pose)
            df_angles_features = update_dataframe(df_angles_features, angles_features)

            visualize_skeleton_and_landmarks(frame, results, mp_pose,mp_drawing)

            process_dbscan_and_visualize(ax,frame, df_coords_features)

        cv2.imshow('Landmark Skeleton', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    release_resources(cap)

def update_dataframe(existing_df, new_data):
    return pd.DataFrame.from_dict([new_data], orient='columns') if existing_df.empty else pd.concat(
        [existing_df, pd.DataFrame([new_data], columns=existing_df.columns)], ignore_index=True)

def visualize_skeleton_and_landmarks(frame, results, mp_pose,mp_drawing):
    mp_skeleton_connections = mp_pose.POSE_CONNECTIONS
    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_skeleton_connections)

def process_dbscan_and_visualize(ax, frame, df_coords_features):
    if df_coords_features.shape[0] > MAX_SAMPLES_FOR_DBSCAN:
        df_coords_features = df_coords_features.drop(0).reset_index(drop=True)

    if df_coords_features.shape[0] > MIN_SAMPLES_FOR_DBSCAN:
        df_split_coords_features = split_3d_columns(df_coords_features[SELECTED_COLUMNS_FOR_DBSCAN])
        clusters = dbscan_clustering(df_split_coords_features.values,
                                     epsilon=EPSILON_FOR_DBSCAN, min_samples=MIN_SAMPLES_FOR_DBSCAN,
                                     pca=pca,scaler=scaler)

        cv2.putText(frame, f"Action: {clusters[-1]}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        command(action=clusters[-1])
        visualize_dbscan_clusters(ax, df_split_coords_features.values, clusters,pca,scaler)

def release_resources(cap):
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()