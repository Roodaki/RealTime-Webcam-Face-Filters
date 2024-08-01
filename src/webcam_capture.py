import cv2
from src.webcam_constants import WEBCAM_INDEX, EXIT_KEY, WINDOW_NAME, FRAME_WAIT_KEY
from src.facial_landmark_detection import detect_facial_landmarks, draw_facial_landmarks


def open_webcam_with_landmark_detection():
    """
    Opens the webcam and starts capturing video frames with facial landmark detection.

    The function captures video from the default webcam, detects facial landmarks,
    draws them on the video frames, and exits when the specified exit key is pressed.
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
        frame_with_landmarks = draw_facial_landmarks(frame, landmarks)

        cv2.imshow(WINDOW_NAME, frame_with_landmarks)

        if cv2.waitKey(FRAME_WAIT_KEY) & 0xFF == ord(EXIT_KEY):
            break

    video_capture.release()
    cv2.destroyAllWindows()
