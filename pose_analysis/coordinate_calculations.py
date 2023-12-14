import pandas as pd

def calculate_relative_3d_coordinates(mp_pose,results, frame, x_quantization=1.0, y_quantization=1.0, z_quantization=1.0):
    """
    Calculates 3D coordinates relative to the body's center of mass for all 33 Mediapipe keypoints.

    Args:
        results: The pose detection results from Mediapipe.
        frame: The original frame captured from the camera.
        x_quantization: Quantization step for X coordinates.
        y_quantization: Quantization step for Y coordinates.
        z_quantization: Quantization step for Z coordinates.

    Returns:
        dict: Dictionary containing landmark names as keys and their respective 3D coordinates as values.
    """
    # Mapping landmark numbers to names
    landmark_names = {
        0: "Nose", 1: "Left Eye Inner", 2: "Left Eye", 3: "Left Eye Outer",
        4: "Right Eye Inner", 5: "Right Eye", 6: "Right Eye Outer",
        7: "Left Ear", 8: "Right Ear",
        9: "Mouth Left", 10: "Mouth Right",
        11: "Left Shoulder", 12: "Right Shoulder",
        13: "Left Elbow", 14: "Right Elbow",
        15: "Left Wrist", 16: "Right Wrist",
        17: "Left Pinky", 18: "Right Pinky",
        19: "Left Index", 20: "Right Index",
        21: "Left Thumb", 22: "Right Thumb",
        23: "Left Hip", 24: "Right Hip",
        25: "Left Knee", 26: "Right Knee",
        27: "Left Ankle", 28: "Right Ankle",
        29: "Left Heel", 30: "Right Heel",
        31: "Left Foot Index", 32: "Right Foot Index"
    }


    # Calculate the center of mass coordinates
    center_of_mass_x = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x +
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2
    center_of_mass_y = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y +
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
    center_of_mass_z = (results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].z +
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].z) / 2

    # Dictionary to store landmark names and their respective 3D coordinates
    coordinates_dict = {}

    # Iterate through each pose landmark
    for i in range(33):
        # Extract the landmark object
        landmark = results.pose_landmarks.landmark[i]

        # Calculate normalized 3D coordinates relative to the center of mass
        x_rel = (landmark.x - center_of_mass_x) * frame.shape[1]
        y_rel = (landmark.y - center_of_mass_y) * frame.shape[0]
        z_rel = (landmark.z - center_of_mass_z) * (frame.shape[1] + frame.shape[0]) / 2

        # Quantize the coordinates
        x_quantized = round(x_rel / x_quantization) * x_quantization
        y_quantized = round(y_rel / y_quantization) * y_quantization
        z_quantized = round(z_rel / z_quantization) * z_quantization

        # Store coordinates in the dictionary
        landmark_name = landmark_names[i]

        # Format the values to two decimal places
        x_rel_formatted = "{:.0f}".format(x_quantized)
        y_rel_formatted = "{:.0f}".format(y_quantized)
        z_rel_formatted = "{:.0f}".format(z_quantized)

        # Store the formatted values in coordinates_dict
        coordinates_dict[landmark_name] = (x_rel_formatted, y_rel_formatted, z_rel_formatted)

        # Display landmark name and relative 3D coordinates on the right side of the frame in green
        text = f"{landmark_name}: ({x_rel_formatted}, {y_rel_formatted}, {z_rel_formatted})"
        text_position = (frame.shape[1] - 250, 30 + 15 * i)  # Adjusted the spacing
        #cv2.putText(frame, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

    return coordinates_dict




def split_3d_columns(df):
    new_data = []

    for column in df.columns:
        is_3d_coordinate_column = df[column].apply(lambda x: isinstance(x, tuple) and len(x) == 3).all()

        if is_3d_coordinate_column:
            # Create new columns for x, y, and z coordinates
            x_col = f'{column}_x'
            y_col = f'{column}_y'
            z_col = f'{column}_z'

            new_data.append((x_col, y_col, z_col))

    # Create a new DataFrame with x, y, and z columns
    new_df = pd.DataFrame()

    for x_col, y_col, z_col in new_data:
        new_df[x_col] = df.apply(lambda row: row[column][0], axis=1)
        new_df[y_col] = df.apply(lambda row: row[column][1], axis=1)
        new_df[z_col] = df.apply(lambda row: row[column][2], axis=1)

    return new_df