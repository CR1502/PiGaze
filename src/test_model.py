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
    model.load_state_dict(torch.load(model_path, weights_only=True))
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
    gaze_threshold = 25  # Define acceptable error in Euclidean distance, can change threshold
    head_pose_threshold = 5  # Define acceptable error in degrees, can change threshold

    gaze_within_threshold = 0
    head_pose_within_threshold = 0


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
        
        # Check if errors are within thresholds
        if np.all(np.abs(gaze_error) <= gaze_threshold):
            gaze_within_threshold += 1
        if np.all(np.abs(head_pose_error) <= head_pose_threshold):  # All angles within the threshold
            head_pose_within_threshold += 1

    gaze_accuracy = (gaze_within_threshold / len(testing_dataset)) * 100
    head_pose_accuracy = (head_pose_within_threshold / len(testing_dataset)) * 100

    head_pose_errors = np.array(head_pose_errors)
    return gaze_errors, head_pose_errors, gaze_accuracy, head_pose_accuracy
        
if __name__ == "__main__":
    model_path = 'models/pigaze_model5(final).pth'
    print("testing: ", model_path)
    gaze_errors, head_pose_errors, gaze_accuracy, head_pose_accuracy = test_model(model_path)
    
    plt.figure(figsize=(12, 6)) 
    plt.plot(range(len(gaze_errors)), gaze_errors, label="Gaze Target Error", color="blue", marker="o")
    plt.title("Gaze Target Error Across Test Samples")
    plt.suptitle(f"Model: {model_path}")
    plt.xlabel("Sample Index")
    plt.ylabel("Gaze Error (Euclidean Distance)")
    plt.xticks(np.arange(0, len(gaze_errors), step=10))  
    plt.legend()
    plt.grid(True)
    # Saving and showing the plot
    plt.savefig(f"{model_path}_gaze_errors_plot.png") 
    plt.show()
    
    print(f"Gaze Accuracy: {gaze_accuracy:.2f}%")
    print(f"Head Pose Accuracy: {head_pose_accuracy:.2f}%")
    