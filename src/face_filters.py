# src/face_filters.py

import cv2
import numpy as np
from src.webcam_constants import BLUR_KERNEL_SIZE


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
