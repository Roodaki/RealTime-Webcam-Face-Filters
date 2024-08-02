import cv2
import numpy as np
from src.webcam_constants import (
    BLUR_KERNEL_SIZE,
    SUNGLASSES_IMAGE_PATH,
    MUSTACHE_IMAGE_PATH,
)


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


def apply_mustache_filter(frame, landmarks):
    """
    Apply a mustache filter to the face using the detected landmarks.

    Args:
        frame (numpy.ndarray): The frame from the webcam capture.
        landmarks (list): A list of facial landmarks.

    Returns:
        frame (numpy.ndarray): The frame with the mustache filter applied.
    """
    if not landmarks:
        return frame

    # Load the mustache image
    mustache = cv2.imread(MUSTACHE_IMAGE_PATH, cv2.IMREAD_UNCHANGED)
    if mustache is None:
        print(f"Error: Unable to load mustache image from {MUSTACHE_IMAGE_PATH}")
        return frame

    for face_landmarks in landmarks:
        # Get the coordinates for the nose and upper lip
        nose_tip = face_landmarks[1]  # Nose tip
        left_mouth_corner = face_landmarks[61]  # Left mouth corner
        right_mouth_corner = face_landmarks[291]  # Right mouth corner

        # Calculate the width and height of the mustache
        mouth_width = int(
            np.linalg.norm(np.array(right_mouth_corner) - np.array(left_mouth_corner))
        )
        mustache_width = int(
            mouth_width * 1.5
        )  # Adjust the multiplier for a better fit
        aspect_ratio = mustache.shape[0] / mustache.shape[1]
        mustache_height = int(mustache_width * aspect_ratio)

        # Resize the mustache image
        resized_mustache = cv2.resize(
            mustache,
            (mustache_width, mustache_height),
            interpolation=cv2.INTER_AREA,
        )

        # Calculate the angle between the mouth corners
        mouth_delta_x = right_mouth_corner[0] - left_mouth_corner[0]
        mouth_delta_y = right_mouth_corner[1] - left_mouth_corner[1]
        angle = -np.degrees(np.arctan2(mouth_delta_y, mouth_delta_x))

        # Rotate the mustache image
        M = cv2.getRotationMatrix2D(
            (mustache_width // 2, mustache_height // 2), angle, 1.0
        )
        rotated_mustache = cv2.warpAffine(
            resized_mustache,
            M,
            (mustache_width, mustache_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Calculate the position to overlay the mustache
        center = np.mean(
            [nose_tip, left_mouth_corner, right_mouth_corner], axis=0
        ).astype(int)
        top_left = (
            int(center[0] - mustache_width / 2),
            int(
                nose_tip[1] - mustache_height * 0.2
            ),  # Adjust vertical position to move the mustache up
        )

        # Ensure the coordinates are within the frame bounds
        top_left_y = max(0, top_left[1])
        bottom_right_y = min(frame.shape[0], top_left[1] + mustache_height)
        top_left_x = max(0, top_left[0])
        bottom_right_x = min(frame.shape[1], top_left[0] + mustache_width)

        # Adjust the region of interest (ROI) in the frame
        roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
        mustache_roi = rotated_mustache[
            top_left_y - top_left[1] : bottom_right_y - top_left[1],
            top_left_x - top_left[0] : bottom_right_x - top_left[0],
        ]

        # Overlay the mustache on the frame
        for i in range(mustache_roi.shape[0]):
            for j in range(mustache_roi.shape[1]):
                if mustache_roi[i, j, 3] > 0:  # Alpha channel check
                    roi[i, j] = mustache_roi[i, j, :3]

    return frame
