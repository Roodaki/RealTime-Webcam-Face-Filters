<div align="center">
  <h1><strong>Webcam-Based Real-Time Face Filters via Facial Landmark Detection</strong></h1>
  
  https://github.com/user-attachments/assets/e0571a98-682a-44ca-b16c-e016a73239b8
</div>

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Intruduction](#intruduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Intruduction

This application applies dynamic filters to your face using a webcam in real time. It leverages facial landmark detection and offers several filters, including facial landmark overlays, blur, sunglasses, and mustache, with live updates based on face orientation.

## Features

1. `Real-Time Face Filters`: Transform your face live using your webcam. The application processes video frames in real time, applying various filters that adapt to your face's movements and orientation.
2. `Facial Landmark Detection`: Utilizes MediaPipe's Face Mesh to detect and overlay facial landmarks accurately. This feature provides a detailed map of facial features, allowing other filters to align perfectly with the face.
3. `Dynamic Filters`:
    - `Blur Filter`: Apply a Gaussian blur to the face region, which softens facial features. The blur intensity adjusts to cover the entire face, ensuring a smooth and consistent effect.
    - `Sunglasses Filter`: Adds a pair of sunglasses to the face that adjusts automatically based on the face's orientation. The sunglasses image resizes and rotates in real-time to fit and align with the eyes, enhancing realism and comfort.
    - `Mustache Filter`: Overlays a mustache on the face that adjusts in size and position according to facial landmarks. The mustache’s placement dynamically adapts to ensure it stays correctly positioned just below the nose, providing a natural appearance.
4. `Responsive Filter Application`: Filters are applied in real-time, responding to changes in facial expressions and movements. The application maintains high performance and responsiveness, ensuring a smooth user experience.

## Project Structure

The project follows a specific structure to organize its files and directories:
```
├── assets/
│   ├── sunglasses.png                 # Image file for the sunglasses filter.
│   └── mustache.png                   # Image file for the mustache filter.
│
├── src/
│   ├── face_filters.py                # Module containing filter application functions.
│   ├── facial_landmark_detection.py   # Module for detecting and drawing facial landmarks.
│   ├── webcam_capture.py              # Module for capturing video from the webcam and applying filters.
│   ├── webcam_constants.py            # File for storing constant values related to webcam and filters.
│   ├── main.py                        # Main script for initializing the webcam and filter application.
│   └── menu.py                        # Module for displaying the on-screen menu.
│
├── requirements.txt                   # Lists the project's dependencies.
├── .gitignore                         # Specifies which files and directories should be ignored by Git version control.
└── README.md                          # Documentation file providing information about the project.
```

## Getting Started

### Requirements
- Python 3.6 or higher
- OpenCV
- MediaPipe
- NumPy

### Installation
1. **Install Python**: Ensure Python 3.X is installed. If not, download and install it from [python.org](https://www.python.org/downloads/).
2. **Clone the Repository**:
   ```
   git clone https://github.com/Roodaki/RealTime-Webcam-Face-Filters
   cd RealTime-Webcam-Face-Filters/
   ```
3. **Install required packages**: Use pip to install the necessary Python libraries:
   ```
   pip install -r requirements.txt
   ```

### Usage
Run the main script to start the application:
```
python main.py
```
Use the following keys to switch between filters:
- `0`: No Filter
- `1`: Facial Landmark Filter
- `2`: Blur Filter
- `3`: Sunglasses Filter
- `4`: Mustache Filter
- `q`: Quit the application

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

For any questions or feedback, please contact amirhossein.rdk@gmail.com.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

