import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2



class MPIIFaceGazeDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        for participant in range(15):  # 15 participants from p00 to p14
            # participant_folder = os.path.join(root_dir, f'p{participant:02d}')
            # annotation_file = os.path.join(participant_folder, f'p{participant:02d}.txt')
            participant_folder = root_dir + f'/p{participant:02d}'
            annotation_file = participant_folder + f'/p{participant:02d}.txt'

            with open(annotation_file, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    image_path = os.path.join(participant_folder, data[0])

                    # Extract relevant information
                    gaze_location = np.array(data[1:3], dtype=float)
                    facial_landmarks = np.array(data[3:15], dtype=float).reshape(-1, 2)
                    head_pose = np.array(data[15:21], dtype=float)
                    face_center = np.array(data[21:24], dtype=float)
                    gaze_target = np.array(data[24:27], dtype=float)

                    self.samples.append({
                        'image_path': image_path,
                        'gaze_location': gaze_location,
                        'facial_landmarks': facial_landmarks,
                        'head_pose': head_pose,
                        'face_center': face_center,
                        'gaze_target': gaze_target
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image_path']).convert('RGB')
        head_pose = sample['head_pose']
        if self.transform:
            image = self.transform(image)

        # Combine gaze target and head pose for the label
        #label = np.concatenate([sample['gaze_location'], sample['head_pose']])
        label = sample['gaze_location']
        return image, torch.tensor(label, dtype=torch.float32)



class PiGazeModel(nn.Module):
    def __init__(self):
        super(PiGazeModel, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # 2 for gaze target, 6 for head pose
        )
        self.c1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.c2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.c3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.dp25 = nn.Dropout(0.25)
        self.dp50 = nn.Dropout(0.5)

    def forward(self, x):

        #x = self.features(xi)
        #x = self.maxpool(x)
        #x = self.avgpool(x)
        x = self.c1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dp50(x)

        x = self.c2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dp50(x)

        x = self.c3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.dp50(x)

        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



# Define image transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets and dataloaders
data_path = r'C:\Users\justi\Documents\NEU\Fall_2024\MLFoundations\GroupProject\dataverse_files\MPIIFaceGaze\Data\Data'
dataset = MPIIFaceGazeDataset(root_dir=data_path, transform=transform)

# Split dataset into train and validation sets
train_size = int(0.9 * len(dataset)) # you an also modify size of the taining set
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)



# Initialize the model
model = PiGazeModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5  # play around with this no.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

# losses = []
# val_losses = []
# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0.0
#     for batch_idx, (i1, i2) in enumerate(train_loader):
#         data, target = i1
#
#         data, target = data.to(device), target.to(device)
#
#         output = model(data)
#
#         loss = criterion(output, target)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()*data.size(0)
#
#         if batch_idx % 50 == 0:
#             print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss / batch_size:.4f}')
#
#
#     # Validation
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for batch_idx, (i1, i2) in enumerate(zip(val_loader, val_loaderH)):
#             data, target = i1
#
#             data, target = data.to(device), target.to(device)
#
#             output = model(data)
#             val_loss += criterion(output, target).item()
#
#
#     print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/batch_size:.4f}, Val Loss: {val_loss/batch_size:.4f}')
#     losses.append(train_loss/batch_size)
#     val_losses.append(val_loss/batch_size)
# # Save the model
# torch.save(model.state_dict(), 'pigaze_model.pth')
#
# fig, (ax1, ax2) = plt.subplots(1,2)
# ax1.plot(losses, label='loss')
# ax2.plot(val_losses, label = 'val_loss')
# ax1.set_xlabel('Epoch')
# ax1.set_ylabel('Loss')
# ax1.legend()
# ax2.set_xlabel('Epoch')
# ax2.set_ylabel('Loss')
# ax2.legend()
# plt.subplots_adjust(wspace=0.5, hspace=0.5)
# fig.savefig("Loss Plots")

"""

The Below Code is to test the trained model

"""
# Function to load and use the model for inference
def load_model(model_path):
    model = PiGazeModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict_gaze(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    gaze_target = output[:, :2].numpy()
    #head_pose = output[:, 2:].numpy()
    #return gaze_target, head_pose
    return gaze_target

# Example usage of the trained model
trained_model = load_model('pigaze_model.pth')
test_image = r'\day02\0659.jpg'
test_image_path = r"C:\Users\justi\Documents\NEU\Fall_2024\MLFoundations\GroupProject\dataverse_files\MPIIFaceGaze\Data\Data\p00" + test_image
test_data_path = r"C:\Users\justi\Documents\NEU\Fall_2024\MLFoundations\GroupProject\dataverse_files\MPIIFaceGaze\Data\Data\p00\p00.txt"
with open(test_data_path, 'r') as f:
    for line in f:
        data = line.strip().split()
        if(data[0] == "day02/0659.jpg"):
            true_gaze_location = np.array(data[1:3], dtype=float)
            true_head_pose = np.array(data[15:21], dtype=float)

#gaze_target, head_pose = predict_gaze(trained_model, test_image_path, transform)
gaze_target = predict_gaze(trained_model, test_image_path, transform,)

print(f"Gaze Target {gaze_target}")
print(f"True Gaze Location {true_gaze_location}")
print(f"Percent Error in X and Y directions: {np.abs((true_gaze_location - gaze_target)/true_gaze_location)}")
#print(f"Head pose Error: {true_head_pose - head_pose}")
cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
image = cv2.imread(r'C:\Users\justi\Documents\NEU\Fall_2024\MLFoundations\GroupProject\PiGaze\Screenshot 2024-11-21 201902.png')


x, y, = gaze_target[0]
x = int(x)
y = int(y)
cv2.circle(image, (x, y), 3, (255, 0, 0), 5)

x, y, = true_gaze_location
x = int(x)
y = int(y)
cv2.circle(image, (x, y), 3, (0, 255, 0), 5)

cv2.imshow("Display", image)
cv2.waitKey(0)
