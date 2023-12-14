import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def draw_connections(ax, pose_landmarks, connections, color="red", thickness=2):
  """
  Draws lines connecting the specified landmarks in 3D space.

  Args:
    ax: Matplotlib 3D axes object.
    pose_landmarks: MediaPipe pose landmarks object.
    connections: A list of tuples representing pairs of landmark indices to connect.
    color: The color of the lines.
    thickness: The thickness of the lines.
  """
  for i, j in connections:
    landmark_i = pose_landmarks.landmark[i]
    landmark_j = pose_landmarks.landmark[j]

    scaled_x_i = landmark_i.x * 0.5 - 0.5
    scaled_y_i = landmark_i.y * 0.5 - 0.5
    scaled_z_i = landmark_i.z * 0.5 - 0.5

    scaled_x_j = landmark_j.x * 0.5 - 0.5
    scaled_y_j = landmark_j.y * 0.5 - 0.5
    scaled_z_j = landmark_j.z * 0.5 - 0.5

    ax.plot3D([scaled_x_i, scaled_x_j], [scaled_y_i, scaled_y_j], [scaled_z_i, scaled_z_j], color=color, linewidth=thickness)


# Create a pose detection object
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the video capture
cap = cv2.VideoCapture(0)

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlim3d(-1, 1)
ax.set_ylim3d(-1, 1)
ax.set_zlim3d(-1, 1)
ax.view_init(elev=15, azim=-60)

while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Convert the frame to RGB
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  # Make detection
  results = pose.process(frame_rgb)

  # Re-convert the frame to BGR
  frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

  # Extract landmarks
  pose_landmarks = results.pose_landmarks

  # Plot landmarks and connections
  ax.clear()
  if pose_landmarks:
    for landmark in pose_landmarks.landmark:
      scaled_x = landmark.x * 0.5 - 0.5
      scaled_y = landmark.y * 0.5 - 0.5
      scaled_z = landmark.z * 0.5 - 0.5
      ax.scatter(scaled_x, scaled_y, scaled_z, color='blue', marker='o', s=20)

    draw_connections(ax, pose_landmarks, mp_pose.POSE_CONNECTIONS)

  # Display frame and 3D plot
  cv2.imshow('MediaPipe Pose', frame)
  plt.draw()
  plt.pause(0.01)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

# Release resources
cap.release()
cv2.destroyAllWindows()
plt.close()
