import cv2
import torch
import torch.nn as nn
import dlib
import numpy as np


class EyeTrackingModel(nn.Module):
    """
    Neural network model for eye tracking that predicts gaze direction 
    based on facial landmarks.
    """
    def __init__(self):
        super(EyeTrackingModel, self).__init__()
        # First fully connected (FC) layer for processing 136 landmarks (68 points * 2 coordinates)
        self.fc_landmarks = nn.Sequential(
            nn.Linear(136, 256),  # Input: 136, Output: 256
            nn.ReLU(),  # Activation function
            nn.Linear(256, 128)  # Output: 128 features
        )
        # Second FC layer for predicting gaze direction
        self.fc_combined = nn.Sequential(
            nn.Linear(128, 64),  # Input: 128, Output: 64
            nn.ReLU(),  # Activation function
            nn.Linear(64, 2)  # Output: 2 (gaze x, y coordinates)
        )

    def forward(self, landmarks):
        """
        Forward pass of the model.

        Args:
        - landmarks (Tensor): Input landmarks (shape: [batch_size, 136]).

        Returns:
        - Tensor: Predicted gaze coordinates (x, y).
        """
        x = self.fc_landmarks(landmarks)  # Process landmarks through the first FC layer
        return self.fc_combined(x)  # Output gaze coordinates


def map_direction(gaze_x, gaze_y):
    """
    Maps gaze coordinates to a direction (e.g., "Left", "Right", "Up", "Down", "Center").

    Args:
    - gaze_x (float): Predicted horizontal gaze coordinate.
    - gaze_y (float): Predicted vertical gaze coordinate.

    Returns:
    - str: Direction of gaze.
    """
    # Define thresholds for center region
    center_x_min, center_x_max = 0.445, 0.5
    center_y_min, center_y_max = 0.389, 0.4

    # Debugging: Print gaze predictions
    print(f"Mapping Gaze: x={gaze_x:.4f}, y={gaze_y:.4f}")

    # Check if the gaze is in the center region
    if center_x_min <= gaze_x <= center_x_max and center_y_min <= gaze_y <= center_y_max:
        print("Direction: Center")
        return "Center"

    # Map gaze to other directions based on thresholds
    if gaze_x < center_x_min:
        print("Direction: Left")
        return "Left"
    elif gaze_x > center_x_max:
        print("Direction: Right")
        return "Right"
    elif gaze_y < center_y_min:
        print("Direction: Up")
        return "Up"
    elif gaze_y > center_y_max:
        print("Direction: Down")
        return "Down"

    # Fallback case for unexpected values
    print("Direction: Undefined")
    return "Undefined"


def test_live_feed():
    """
    Tests the eye-tracking model using live webcam feed, displays direction, 
    and saves the video feed in MP4 format.
    """
    # Load the trained model from the saved state
    model = EyeTrackingModel()
    model.load_state_dict(torch.load("model_9.pth", map_location=torch.device("cpu")))
    model.eval()  # Set the model to evaluation mode

    # Initialize Dlib's face detector and shape predictor
    detector = dlib.get_frontal_face_detector()  # Detects faces in images
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Predicts facial landmarks

    # Open webcam feed
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set frame width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set frame height

    # Define the codec and initialize the VideoWriter for MP4 format
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter('eye_tracking_output_1.mp4', fourcc, 20.0, (640, 480))  # Save output to file

    # Check if the webcam feed is opened successfully
    if not cap.isOpened():
        print("Failed to open camera.")
        return

    while True:
        # Read frame from the webcam
        ret, frame = cap.read()
        if not ret:  # Break the loop if no frame is captured
            break

        # Flip the frame horizontally to create a mirror view
        frame = cv2.flip(frame, 1)

        # Convert the frame to grayscale for Dlib face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)  # Detect faces in the frame

        if len(faces) > 0:  # If at least one face is detected
            for face in faces:
                # Extract facial landmarks for the detected face
                landmarks = predictor(gray, face)
                landmark_coords = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32).flatten()
                landmark_tensor = torch.tensor(landmark_coords, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

                # Predict gaze direction using the trained model
                with torch.no_grad():  # Disable gradient computation
                    gaze = model(landmark_tensor)
                    gaze_x, gaze_y = gaze[0].numpy()  # Extract gaze coordinates

                # Debugging: Print gaze coordinates
                print(f"Gaze Coordinates: x={gaze_x:.4f}, y={gaze_y:.4f}")

                # Map the predicted gaze to a direction
                direction = map_direction(gaze_x, gaze_y)

                # Display the direction on the frame
                cv2.putText(frame, f"Direction: {direction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        else:
            # If no face is detected, display a message on the frame
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Write the processed frame to the output video file
        out.write(frame)

        # Display the frame in a window
        cv2.imshow("Eye Tracking", frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the webcam and video writer resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()  # Close all OpenCV windows


# Run the live feed test
test_live_feed()