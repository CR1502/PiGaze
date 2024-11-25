import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import MPII
import matplotlib.pyplot as plt

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
    head_pose = output[:, 2:].numpy()
    return gaze_target, head_pose

def test_model(model_path):
    trained_model = load_model(model_path)

    data_path = os.path.join(os.getcwd(), 'data', 'dataset', 'dataverse_files', 'MPIIFaceGaze', 'Data', 'Data')
    # Make sure the dataset folder exists
    if not os.path.exists(data_path):
        print("Dataset folder not found. Please place the dataset in 'data/dataset/'.")

    testing_dataset = MPII.MPIIFaceGazeDataset(root_dir=data_path, transform=MPII.transform, participants=range(11, 15))  # p11 to p14

    gaze_errors = []
    head_pose_errors = []

    for i, sample in enumerate(testing_dataset):
        # if i == 4:
        #     break
        gaze_target, head_pose = predict_gaze(trained_model, sample['image_path'], MPII.transform)
        true_gaze_location = sample['gaze_location']
        true_head_pose = sample['head_pose']
        
        gaze_error = np.linalg.norm(true_gaze_location - gaze_target[0]) #Euclidean distance
        head_pose_error = np.abs(true_head_pose - head_pose[0])
        
        gaze_errors.append(gaze_error)
        head_pose_errors.append(head_pose_error)

    head_pose_errors = np.array(head_pose_errors)
    return gaze_errors, head_pose_errors


        
if __name__ == "__main__":
    model_path = 'models/pigaze_model5(final).pth'
    gaze_errors, head_pose_errors = test_model(model_path)
    
    plt.figure(figsize=(12, 6)) 
    plt.plot(range(len(gaze_errors)), gaze_errors, label="Gaze Target Error", color="blue", marker="o")
    plt.title("Gaze Target Error Across Test Samples")
    plt.xlabel("Sample Index")
    plt.ylabel("Gaze Error (Euclidean Distance)")
    plt.xticks(np.arange(0, len(gaze_errors), step=10))  
    plt.legend()
    plt.grid(True)
    # Saving and showing the plot
    plt.savefig("gaze_error_plot.png") 
    plt.show()
    
# test_image_path = r'C:\Users\justi\Documents\NEU\Fall_2024\MLFoundations\GroupProject\dataverse_files\MPIIFaceGaze\Data\Data\p00\day01\0005.jpg'
# test_data_path = r"C:\Users\justi\Documents\NEU\Fall_2024\MLFoundations\GroupProject\dataverse_files\MPIIFaceGaze\Data\Data\p00\p00.txt"
# with open(test_data_path, 'r') as f:
#     for line in f:
#         data = line.strip().split()
#         true_gaze_location = np.array(data[1:3], dtype=float)
#         true_head_pose = np.array(data[15:21], dtype=float)

# gaze_target, head_pose = predict_gaze(trained_model, test_image_path, transform)

# #It seems like the x and y of the gaze target got flipped becuase the real values are 476 758 and my pred values were 742.6, 456.3 don't have time to look into right now
# print(f"Gaze Target {gaze_target}")
# print(f"Gaze Target Error: {true_gaze_location - gaze_target}")
# print(f"Head pose Error: {true_head_pose}")