import cv2
import torch
import torch.nn as nn
import dlib
import numpy as np


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
    The regions are hardcoded for simplicity.
    """

    # Hardcoded regions for directions
    left_region = gaze_x < 0.40
    right_region = gaze_x > 0.55
    up_region = gaze_y < 0.40
    down_region = gaze_y > 0.50
    center_region_x = 0.40 <= gaze_x <= 0.45
    center_region_y = 0.40 <= gaze_y <= 0.45

    # Determine direction
    if center_region_x and center_region_y:
        return "Center"
    elif left_region:
        return "Left"
    elif right_region:
        return "Right"
    elif up_region:
        return "Up"
    elif down_region:
        return "Down"



def test_live_feed():
    """
    Runs the live test for eye tracking, displaying direction only (no gaze dot).
    """
    # Load the trained model
    model = EyeTrackingModel()
    model.load_state_dict(torch.load("model_9.pth", map_location=torch.device("cpu")))
    model.eval()

    # Initialize Dlib's face detector and shape predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # Open webcam feed
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Failed to open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip the frame (mirror view)
        frame = cv2.flip(frame, 1)

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

                # Debugging: Print gaze coordinates
                print(f"Gaze Coordinates: x={gaze_x:.4f}, y={gaze_y:.4f}")

                # Map gaze to direction
                direction = map_direction(gaze_x, gaze_y)

                # Draw direction on the frame
                cv2.putText(frame, f"Direction: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            # If no face detected
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Eye Tracking", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# Run the live feed test
test_live_feed()