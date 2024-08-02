import cv2
from src.webcam_constants import (
    WEBCAM_INDEX,
    EXIT_KEY,
    WINDOW_NAME,
    FRAME_WAIT_KEY,
    FILTER_NONE_KEY,
    FILTER_LANDMARK_KEY,
    FILTER_BLUR_KEY,
    FILTER_SUNGLASSES_KEY,
)
from src.facial_landmark_detection import detect_facial_landmarks, draw_facial_landmarks
from src.face_filters import apply_blur_filter, apply_sunglasses_filter


def open_webcam_with_filter_switching():
    """
    Opens the webcam and starts capturing video frames with real-time filter switching.

    The function captures video from the default webcam, allows the user to switch
    between plain, facial landmark detection, blur filter, and sunglasses filter in real-time,
    and exits when the specified exit key is pressed.
    """
    video_capture = cv2.VideoCapture(WEBCAM_INDEX)
    if not video_capture.isOpened():
        print(f"Error: Unable to access the webcam at index {WEBCAM_INDEX}")
        return

    current_filter = FILTER_NONE_KEY

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to read frame from webcam")
            break

        # Apply the selected filter
        if current_filter == FILTER_LANDMARK_KEY:
            landmarks = detect_facial_landmarks(frame)
            frame = draw_facial_landmarks(frame, landmarks)
        elif current_filter == FILTER_BLUR_KEY:
            landmarks = detect_facial_landmarks(frame)
            frame = apply_blur_filter(frame, landmarks)
        elif current_filter == FILTER_SUNGLASSES_KEY:
            landmarks = detect_facial_landmarks(frame)
            frame = apply_sunglasses_filter(frame, landmarks)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(FRAME_WAIT_KEY) & 0xFF
        if key == ord(EXIT_KEY):
            break
        elif key == ord(FILTER_NONE_KEY):
            current_filter = FILTER_NONE_KEY
        elif key == ord(FILTER_LANDMARK_KEY):
            current_filter = FILTER_LANDMARK_KEY
        elif key == ord(FILTER_BLUR_KEY):
            current_filter = FILTER_BLUR_KEY
        elif key == ord(FILTER_SUNGLASSES_KEY):
            current_filter = FILTER_SUNGLASSES_KEY

    video_capture.release()
    cv2.destroyAllWindows()
