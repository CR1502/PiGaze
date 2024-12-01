import cv2
import torch
import torch.nn as nn
import dlib
import numpy as np
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

class EyeTrackingModel(nn.Module):
    def __init__(self):
        super(EyeTrackingModel, self).__init__()
        self.fc_landmarks = nn.Sequential(
            nn.Linear(136, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.fc_combined = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Predict gaze point (x, y)
        )

    def forward(self, landmarks):
        x = self.fc_landmarks(landmarks)
        return self.fc_combined(x)


def map_direction(gaze_x, gaze_y):
    """
    Maps gaze coordinates to a direction (e.g., "Left", "Right", "Up", "Down", "Center").
    """
    center_x_min, center_x_max = 0.42, 0.50
    center_y_min, center_y_max = 0.42, 0.50

    if center_x_min <= gaze_x <= center_x_max and center_y_min <= gaze_y <= center_y_max:
        return "Center"

    if gaze_x < center_x_min:
        return "Left"
    elif gaze_x > center_x_max:
        return "Right"
    elif gaze_y < center_y_min:
        return "Up"
    elif gaze_y > center_y_max:
        return "Down"

    return "Undefined"


def test_live_feed():
    """
    Runs the live test for eye tracking on a Raspberry Pi with Pi Camera.
    Saves the live feed to a video file.
    """
    # Load the trained model
    model = EyeTrackingModel()
    model.load_state_dict(torch.load("model_9.pth", map_location=torch.device("cpu")))
    model.eval()

    # Initialize Dlib's face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Initialize Pi Camera
    picam2 = Picamera2()
    video_config = picam2.create_video_configuration(main={"size": (640, 480)})
    picam2.configure(video_config)

    # Video saving setup
    output_path = "gaze_tracking_output_rpi.h264"
    encoder = MJPEGEncoder()
    output = FileOutput(output_path)

    picam2.start_recording(encoder, output)
    picam2.start()

    try:
        while True:
            # Capture frame from Pi Camera
            frame = picam2.capture_array()

            # Convert frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            if len(faces) > 0:
                for face in faces:
                    # Extract landmarks
                    landmarks = predictor(gray, face)
                    landmark_coords = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32).flatten()
                    landmark_tensor = torch.tensor(landmark_coords, dtype=torch.float32).unsqueeze(0)

                    # Predict gaze
                    with torch.no_grad():
                        gaze = model(landmark_tensor)
                        gaze_x, gaze_y = gaze[0].numpy()

                    # Map gaze to direction
                    direction = map_direction(gaze_x, gaze_y)

                    # Display direction on the frame
                    cv2.putText(frame, f"Direction: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            else:
                # If no face detected
                cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Show the live feed on screen
            cv2.imshow("Eye Tracking", frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Exiting...")
                break

    finally:
        # Ensure resources are released properly
        picam2.stop_recording()
        picam2.stop()
        cv2.destroyAllWindows()


# Run the test
test_live_feed()
