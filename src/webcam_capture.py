# src/webcam_capture.py

import cv2
from src.webcam_constants import WEBCAM_INDEX, EXIT_KEY, WINDOW_NAME, FRAME_WAIT_KEY
from src.facial_landmark_detection import detect_facial_landmarks
from src.face_filters import apply_blur_filter


def open_webcam_with_filters():
    """
    Opens the webcam and starts capturing video frames with facial filters applied.

    The function captures video from the default webcam, detects facial landmarks,
    applies filters to the face, and exits when the specified exit key is pressed.
    """
    video_capture = cv2.VideoCapture(WEBCAM_INDEX)
    if not video_capture.isOpened():
        print(f"Error: Unable to access the webcam at index {WEBCAM_INDEX}")
        return

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to read frame from webcam")
            break

        landmarks = detect_facial_landmarks(frame)
        frame_with_filter = apply_blur_filter(frame, landmarks)

        cv2.imshow(WINDOW_NAME, frame_with_filter)

        if cv2.waitKey(FRAME_WAIT_KEY) & 0xFF == ord(EXIT_KEY):
            break

    video_capture.release()
    cv2.destroyAllWindows()
