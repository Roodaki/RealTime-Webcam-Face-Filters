# src/main.py

from src.webcam_capture import open_webcam_with_filters


def main():
    """
    Main function to start the webcam capture with facial filters.

    This function acts as the entry point of the program and starts the webcam capture
    with facial filters by calling the open_webcam_with_filters function.
    """
    open_webcam_with_filters()


if __name__ == "__main__":
    main()
