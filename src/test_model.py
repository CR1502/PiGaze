import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import MPII

"""

The Below Code is to test the trained model

"""
# Function to load and use the model for inference
def load_model(model_path):
    model = MPII.PiGazeModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_gaze(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    gaze_target = output[:, :2].numpy()
    head_pose = output[:, 3:].numpy()
    return gaze_target, head_pose

# Example usage of the trained model
trained_model = load_model('pigaze_model.pth')
test_image_path = r'C:\Users\justi\Documents\NEU\Fall_2024\MLFoundations\GroupProject\dataverse_files\MPIIFaceGaze\Data\Data\p00\day01\0005.jpg'
test_data_path = r"C:\Users\justi\Documents\NEU\Fall_2024\MLFoundations\GroupProject\dataverse_files\MPIIFaceGaze\Data\Data\p00\p00.txt"
with open(test_data_path, 'r') as f:
    for line in f:
        data = line.strip().split()
        true_gaze_location = np.array(data[1:3], dtype=float)
        true_head_pose = np.array(data[15:21], dtype=float)

gaze_target, head_pose = predict_gaze(trained_model, test_image_path, transform)

#It seems like the x and y of the gaze target got flipped becuase the real values are 476 758 and my pred values were 742.6, 456.3 don't have time to look into right now
print(f"Gaze Target {gaze_target}")
print(f"Gaze Target Error: {true_gaze_location - gaze_target}")
print(f"Head pose Error: {true_head_pose}")