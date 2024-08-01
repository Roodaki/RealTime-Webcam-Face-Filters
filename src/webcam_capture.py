import cv2
from src.webcam_constants import WEBCAM_INDEX, EXIT_KEY, WINDOW_NAME, FRAME_WAIT_KEY


def open_webcam_capture():
    """
    Opens the webcam and starts capturing video frames.

    The function captures video from the default webcam, displays the video frames in a window, and exits
    when the specified exit key is pressed.
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

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(FRAME_WAIT_KEY) & 0xFF == ord(EXIT_KEY):
            break

    video_capture.release()
    cv2.destroyAllWindows()
