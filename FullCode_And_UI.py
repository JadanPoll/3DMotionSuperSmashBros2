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
import joblib  # Import joblib to save/load models
from tkinter import filedialog
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pose_analysis import *  # Import your functions and classes
from datetime import datetime
# Constants
MIN_SAMPLES_FOR_DBSCAN = 50
MAX_SAMPLES_FOR_DBSCAN = 500
EPSILON_FOR_DBSCAN = 0.8
SELECTED_COLUMNS_FOR_DBSCAN = ["Nose"]

class MyApp:
    def __init__(self, root):



        self.root = root
        self.root.title("Pose Analysis App")

        self.cap = cv2.VideoCapture(0)
        self.video_label = ttk.Label(root)
        self.video_label.pack()

        self.start_button = ttk.Button(root, text="Start Analysis", command=self.start_analysis)
        self.start_button.pack()

        self.stop_button = ttk.Button(root, text="Stop Analysis", command=self.stop_analysis)
        self.stop_button.pack()

        self.save_button = ttk.Button(root, text="Save Models", command=self.save_models)
        self.save_button.pack()

        self.load_button = ttk.Button(root, text="Load Models", command=self.load_models)
        self.load_button.pack()
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose()
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize DataFrames
        self.df_coords_features = pd.DataFrame()
        self.df_angles_features = pd.DataFrame()

        self.fig, self.ax = plt.subplots(figsize=(8, 6))

        self.pca = PCA(n_components=2)
        self.scaler = StandardScaler()

    def start_analysis(self):
        self.running = True
        self.analyze_video()

    def stop_analysis(self):
        self.running = False
        self.release_resources(self.cap)


    def save_models(self):
        now = datetime.now().strftime("%Y%m%d%H%M%S")

        # Save PCA and StandardScaler models
        pca_file_path = f"pca_model_{now}.joblib"
        scaler_file_path = f"scaler_model_{now}.joblib"
        joblib.dump(self.pca, pca_file_path)
        joblib.dump(self.scaler, scaler_file_path)

        # Save the trained DBSCAN model
        if self.cluster_model:
            cluster_model_file_path = f"cluster_model_{now}_{'_'.join(SELECTED_COLUMNS_FOR_DBSCAN)}.joblib"
            joblib.dump(self.cluster_model, cluster_model_file_path)

        messagebox.showinfo("Saved", "Models saved successfully!")

    def load_models(self):
        # Use filedialog to open a dialog for choosing files
        pca_file_path = filedialog.askopenfilename(title="Choose PCA Model", filetypes=[("Joblib files", "*.joblib")])
        scaler_file_path = filedialog.askopenfilename(title="Choose Scaler Model", filetypes=[("Joblib files", "*.joblib")])
        cluster_model_file_path = filedialog.askopenfilename(title="Choose Cluster Model", filetypes=[("Joblib files", "*.joblib")])

        try:
            # Load PCA and StandardScaler models
            self.pca = joblib.load(pca_file_path)
            self.scaler = joblib.load(scaler_file_path)

            # Load the trained DBSCAN model
            if cluster_model_file_path:
                self.cluster_model = joblib.load(cluster_model_file_path)

            messagebox.showinfo("Loaded", "Models loaded successfully!")
        except FileNotFoundError:
            messagebox.showerror("Error", "Models not found. Train models before loading.")

    def analyze_video(self):
        _, frame = self.cap.read()

        if frame is not None:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(frame_rgb)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                rel_coords_features = calculate_relative_3d_coordinates(self.mp_pose, results, frame, x_quantization=1.0,
                                                                        y_quantization=1.0, z_quantization=1.0)

                self.df_coords_features = self.update_dataframe(self.df_coords_features, rel_coords_features)

                angles_features = calculate_pose_angles(frame, landmarks, self.mp_pose)
                self.df_angles_features = self.update_dataframe(self.df_angles_features, angles_features)

                self.visualize_skeleton_and_landmarks(frame, results)

                self.process_dbscan_and_visualize(self.ax, frame, self.df_coords_features)

                if self.running:
                    self.show_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    self.root.after(10, self.analyze_video)

    def visualize_skeleton_and_landmarks(self, frame, results):
        mp_skeleton_connections = self.mp_pose.POSE_CONNECTIONS
        self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_skeleton_connections)

    def update_dataframe(self,existing_df, new_data):
        return pd.DataFrame.from_dict([new_data], orient='columns') if existing_df.empty else pd.concat(
            [existing_df, pd.DataFrame([new_data], columns=existing_df.columns)], ignore_index=True)


    def process_dbscan_and_visualize(self,ax, frame, df_coords_features):
        df_split_coords_features = pd.DataFrame()   # Define df_split_coords_features outside the if block
        clusters = []
        if df_coords_features.shape[0] > MAX_SAMPLES_FOR_DBSCAN:
            df_coords_features = df_coords_features.drop(0).reset_index(drop=True)

        if df_coords_features.shape[0] > MIN_SAMPLES_FOR_DBSCAN:
            df_split_coords_features = split_3d_columns(df_coords_features[SELECTED_COLUMNS_FOR_DBSCAN])
            clusters, self.pca, self.scaler,self.cluster_model = dbscan_clustering(df_split_coords_features.values,
                                        epsilon=EPSILON_FOR_DBSCAN, min_samples=MIN_SAMPLES_FOR_DBSCAN,
                                        pca=self.pca,scaler=self.scaler)

            cv2.putText(frame, f"Action: {clusters[-1]}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

            command(action=clusters[-1])
            visualize_dbscan_clusters(self.ax, df_split_coords_features.values, clusters,self.pca,self.scaler)

    def release_resources(self,cap):
        cap.release()
        cv2.destroyAllWindows()

    def show_frame(self, frame):
        frame = cv2.resize(frame, (640, 480))
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

if __name__ == "__main__":
    root = tk.Tk()
    app = MyApp(root)
    root.mainloop()
