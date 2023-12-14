import math
import numpy as np
def calculate_angle(a, b, c):
    radians = math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(math.degrees(radians))
    return angle




def calculate_pose_angles(frame, landmarks,mp_pose):
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


    return angles_dict

       