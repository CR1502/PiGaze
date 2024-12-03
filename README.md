# PiGaze: Real-Time Eye Movement Tracking with Raspberry Pi

PiGaze is a lightweight, real-time eye movement tracking system using a **Raspberry Pi**, **Pi Camera**, and **PyTorch-based deep learning models**. It employs both **CNN** and **FCNN models** for accurate gaze estimation and direction prediction, making it suitable for various interactive applications.

---

## Features

- **Real-Time Gaze Tracking:** Tracks eye movement in real-time.
- **Direction Mapping:** Detects gaze directions (Left, Right, Up, Down, Center).
- **Lightweight Models:** Optimized for Raspberry Pi's limited computational resources.
- **Video Recording:** Saves live-feed eye tracking output in video format.

---

## Requirements

### Hardware
- Raspberry Pi 3/4
- Pi Camera Module

### Software
- Raspberry Pi OS
- Python 3.7+
- PyTorch, OpenCV, Dlib, Picamera2


## Dataset

PiGaze uses the **MPIIFaceGaze Dataset** for training and evaluation. This dataset provides extensive facial images annotated with gaze information, making it suitable for robust gaze estimation models.

### Dataset Overview
- **Dataset Name:** MPIIFaceGaze
- **Annotations:** Includes gaze directions and facial landmarks for multiple participants.
- **Applications:** Gaze estimation, eye-tracking, and head-pose analysis.

### Download Instructions:
1. Visit the [MPIIFaceGaze Dataset page](https://doi.org/10.18419/darus-3240).
2. Click "Access Dataset" and then "Download ZIP."
3. Accept the dataset terms and conditions, then download the file.
4. Extract the dataset into the `data/` directory of this repository.

### Additional Required Files:
- **Facial Landmarks Model:**  
  Download `shape_predictor_68_face_landmarks.dat` from [this GitHub link](https://github.com/italojs/facial-landmarks-recognition/blob/master/shape_predictor_68_face_landmarks.dat) and place it in the working directory.
