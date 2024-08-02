import cv2
import numpy as np
from src.webcam_constants import BLUR_KERNEL_SIZE, SUNGLASSES_IMAGE_PATH


def apply_blur_filter(frame, landmarks):
    """
    Apply a blur filter to the face using the detected landmarks.

    Args:
        frame (numpy.ndarray): The frame from the webcam capture.
        landmarks (list): A list of facial landmarks.

    Returns:
        frame (numpy.ndarray): The frame with the face blurred.
    """
    if not landmarks:
        return frame

    # Create a mask for the face
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    for face_landmarks in landmarks:
        hull = cv2.convexHull(np.array(face_landmarks))
        cv2.fillConvexPoly(mask, hull, 255)

    # Apply the blur to the face region
    blurred_frame = cv2.GaussianBlur(frame, BLUR_KERNEL_SIZE, 0)
    frame = np.where(mask[:, :, np.newaxis] == 255, blurred_frame, frame)

    return frame


def apply_sunglasses_filter(frame, landmarks):
    """
    Apply a sunglasses filter to the face using the detected landmarks.

    Args:
        frame (numpy.ndarray): The frame from the webcam capture.
        landmarks (list): A list of facial landmarks.

    Returns:
        frame (numpy.ndarray): The frame with the sunglasses filter applied.
    """
    if not landmarks:
        return frame

    # Load the sunglasses image
    sunglasses = cv2.imread(SUNGLASSES_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if sunglasses is None:
        print(f"Error: Unable to load sunglasses image from {SUNGLASSES_IMAGE_PATH}")
        return frame

    for face_landmarks in landmarks:
        # Get the coordinates for the eyes
        left_eye = face_landmarks[33]  # Left eye corner
        right_eye = face_landmarks[263]  # Right eye corner

        # Calculate the width and height of the sunglasses
        eye_width = int(np.linalg.norm(np.array(right_eye) - np.array(left_eye)))
        sunglasses_width = int(
            eye_width * 2.2
        )  # Adjust the multiplier for a better fit
        aspect_ratio = sunglasses.shape[0] / sunglasses.shape[1]
        sunglasses_height = int(sunglasses_width * aspect_ratio)

        # Resize the sunglasses image
        resized_sunglasses = cv2.resize(
            sunglasses,
            (sunglasses_width, sunglasses_height),
            interpolation=cv2.INTER_AREA,
        )

        # Calculate the angle between the eyes (invert the sign for correct direction)
        eye_delta_x = right_eye[0] - left_eye[0]
        eye_delta_y = right_eye[1] - left_eye[1]
        angle = -np.degrees(np.arctan2(eye_delta_y, eye_delta_x))  # Inverted sign

        # Rotate the sunglasses image
        M = cv2.getRotationMatrix2D(
            (sunglasses_width // 2, sunglasses_height // 2), angle, 1.0
        )
        rotated_sunglasses = cv2.warpAffine(
            resized_sunglasses,
            M,
            (sunglasses_width, sunglasses_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Calculate the position to overlay the sunglasses
        center = np.mean([left_eye, right_eye], axis=0).astype(int)
        top_left = (
            int(center[0] - sunglasses_width / 2),
            int(center[1] - sunglasses_height / 2),
        )

        # Ensure the coordinates are within the frame bounds
        top_left_y = max(0, top_left[1])
        bottom_right_y = min(frame.shape[0], top_left[1] + sunglasses_height)
        top_left_x = max(0, top_left[0])
        bottom_right_x = min(frame.shape[1], top_left[0] + sunglasses_width)

        # Adjust the region of interest (ROI) in the frame
        roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        sunglasses_roi = rotated_sunglasses[
            top_left_y - top_left[1] : bottom_right_y - top_left[1],
            top_left_x - top_left[0] : bottom_right_x - top_left[0],
        ]

        # Overlay the sunglasses on the frame
        for i in range(sunglasses_roi.shape[0]):
            for j in range(sunglasses_roi.shape[1]):
                if sunglasses_roi[i, j, 3] > 0:  # Alpha channel check
                    roi[i, j] = sunglasses_roi[i, j, :3]

    return frame
