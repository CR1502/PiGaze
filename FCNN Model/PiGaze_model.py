import os
import dlib
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import random

# Dataset for Eye Tracking
class EyeTrackingDataset(Dataset):
    """
    A custom Dataset class for loading eye-tracking data. It processes images, extracts facial landmarks, 
    and associates them with calibration points (gaze target).
    """
    def __init__(self, image_dir, calibration_points):
        """
        Initializes the dataset.

        Args:
        - image_dir (str): Directory containing all images.
        - calibration_points (list): List of gaze points for calibration.

        The dataset uses dlib to detect faces and extract landmarks from the images. Only images with a valid 
        detected face are included in the dataset.
        """
        self.image_dir = image_dir
        self.samples = []  # Store paths to valid images
        self.calibration_points = calibration_points  # Gaze calibration points (e.g., screen coordinates)

        # Initialize dlib's face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()  # Detects faces in images
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Predicts facial landmarks

        # Iterate through image directory to collect valid image paths
        for img_file in sorted(os.listdir(image_dir)):
            if img_file.endswith(".jpg"):  # Consider only JPG images
                image_path = os.path.join(image_dir, img_file)
                # Add to dataset only if a face is detected
                if self.validate_face(image_path):
                    self.samples.append(image_path)

    def validate_face(self, image_path):
        """
        Validates if a face can be detected in the given image.

        Args:
        - image_path (str): Path to the image.

        Returns:
        - bool: True if at least one face is detected, False otherwise.
        """
        image = cv2.imread(image_path)  # Load image
        if image is None:  # Handle case where image could not be loaded
            return False
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert image to grayscale
        faces = self.detector(gray)  # Detect faces
        return len(faces) > 0  # Return True if at least one face is detected

    def __len__(self):
        """
        Returns the number of valid samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches an image, extracts landmarks, and associates them with a calibration point.

        Args:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - torch.Tensor: Flattened landmarks (68 landmarks with x, y coordinates -> 136 values).
        - torch.Tensor: Gaze calibration target (x, y coordinates).
        """
        # Get the image path
        image_path = self.samples[idx]
        image = cv2.imread(image_path)  # Load the image
        if image is None:  # Handle case where the image could not be loaded
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert to grayscale and detect face
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        if len(faces) == 0:  # Handle case where no face is detected
            raise ValueError(f"No face detected in image: {image_path}")

        # Extract landmarks from the detected face
        landmarks = self.predictor(gray, faces[0])  # Use the first detected face
        landmark_coords = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32).flatten()

        # Simulate a gaze target using calibration points (cyclic assignment)
        label = self.calibration_points[idx % len(self.calibration_points)]

        # Convert landmarks and labels to PyTorch tensors
        return torch.tensor(landmark_coords, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)


# Eye Tracking Model
class EyeTrackingModel(nn.Module):
    """
    A simple neural network model to predict gaze coordinates based on facial landmarks.
    """
    def __init__(self):
        """
        Initializes the model.
        The model consists of fully connected (FC) layers to process landmarks and predict gaze coordinates.
        """
        super(EyeTrackingModel, self).__init__()
        # FC layers for processing 136-dimensional landmark input
        self.fc_landmarks = nn.Sequential(
            nn.Linear(136, 256),  # Input: 136 (68 landmarks * 2 coordinates), Output: 256
            nn.ReLU(),
            nn.Linear(256, 128)  # Output: 128 features
        )
        # FC layers for predicting gaze coordinates
        self.fc_combined = nn.Sequential(
            nn.Linear(128, 64),  # Input: 128 features, Output: 64
            nn.ReLU(),
            nn.Linear(64, 2)  # Output: 2 (gaze x, y)
        )

    def forward(self, landmarks):
        """
        Forward pass through the network.

        Args:
        - landmarks (Tensor): Input landmarks (shape: [batch_size, 136]).

        Returns:
        - Tensor: Predicted gaze coordinates (x, y).
        """
        x = self.fc_landmarks(landmarks)  # Process landmarks
        return self.fc_combined(x)  # Predict gaze coordinates


# Calibration points used as gaze targets for training
calibration_points = [
    [0.3, 0.3],  # Top-left corner of the screen
    [0.5, 0.3],  # Top-right corner of the screen
    [0.5, 0.8],  # Bottom-left corner of the screen
    [0.8, 0.8]   # Bottom-right corner of the screen
]

# Create the dataset and split it into training and validation sets
image_dir = "Data_3"  # Path to images
dataset = EyeTrackingDataset(image_dir, calibration_points)  # Initialize dataset
train_size = int(0.8 * len(dataset))  # 80% training
val_size = len(dataset) - train_size  # 20% validation
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders for batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

# Configure the device for computation
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # MPS for macOS or fallback to CPU
print(f"Using device: {device}")

# Initialize the model, loss function, and optimizer
model = EyeTrackingModel().to(device)
criterion = nn.MSELoss()  # Mean Squared Error loss for regression
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer for training

# Training loop
num_epochs = 20
training_losses = []  # List to store training losses
validation_losses = []  # List to store validation losses

for epoch in range(num_epochs):
    # Training phase
    model.train()  # Set model to training mode
    running_loss = 0.0
    running_mae = 0.0  # Mean Absolute Error accumulator

    for landmarks, label in train_loader:
        # Move data to the configured device
        landmarks, label = landmarks.to(device), label.to(device)

        # Forward pass
        optimizer.zero_grad()  # Clear gradients
        predictions = model(landmarks)  # Make predictions

        # Compute loss
        loss = criterion(predictions, label)
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model parameters

        # Update metrics
        running_loss += loss.item()
        mae = torch.mean(torch.abs(predictions - label)).item()  # Calculate MAE
        running_mae += mae

    # Average training loss and MAE for the epoch
    avg_loss = running_loss / len(train_loader)
    avg_mae = running_mae / len(train_loader)
    training_losses.append(avg_loss)  # Store training loss

    # Validation phase
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0
    with torch.no_grad():  # Disable gradient computation
        for landmarks, label in val_loader:
            landmarks, label = landmarks.to(device), label.to(device)
            predictions = model(landmarks)
            val_loss += criterion(predictions, label).item()  # Compute validation loss
    validation_losses.append(val_loss / len(val_loader))  # Store validation loss

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, MAE: {avg_mae:.4f}, Val Loss: {validation_losses[-1]:.4f}")

# Save the trained model
torch.save(model.state_dict(), "model_9.pth")
print("Model saved!")

# Plot Training and Validation Loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs + 1), training_losses, label="Training Loss")
plt.plot(range(1, num_epochs + 1), validation_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid()
plt.show()


# Visualize Landmarks on a Random Image
def visualize_landmarks(dataset, index=None):
    """
    Visualizes facial landmarks on an image from the dataset.

    Args:
    - dataset: EyeTrackingDataset object.
    - index: Optional; the index of the image to visualize. If None, a random index is selected.
    """
    if index is None:
        index = random.randint(0, len(dataset) - 1)  # Randomly select an image index

    # Load the image and detect landmarks
    image_path = dataset.samples[index]
    image = cv2.imread(image_path)  # Read the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = dataset.detector(gray)  # Detect faces
    if len(faces) == 0:  # Handle case where no face is detected
        print(f"No face detected in image: {image_path}")
        return

    # Extract landmarks
    landmarks = dataset.predictor(gray, faces[0])
    landmark_coords = np.array([[p.x, p.y] for p in landmarks.parts()], dtype=np.float32)

    # Plot the image with landmarks
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(landmark_coords[:, 0], landmark_coords[:, 1], c="red", s=20, label="Landmarks")
    plt.title(f"Landmarks on Image: {os.path.basename(image_path)}")
    plt.legend()
    plt.axis("off")
    plt.show()


# Visualize landmarks for a random image
visualize_landmarks(dataset)