import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

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

        if self.transform:
            image = self.transform(image)

        # Combine gaze target and head pose for the label
        label = np.concatenate([sample['gaze_location'], sample['head_pose']])
        return image, torch.tensor(label, dtype=torch.float32)

class PiGazeModel(nn.Module):
    def __init__(self):
        super(PiGazeModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 8)  # 2 for gaze target, 6 for head pose
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
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
data_path = os.path.join(os.getcwd(), 'data', 'dataset',  'dataverser_files', 'MPIIFaceGaze', 'Data', 'Data')
# data_path = r'C:\Users\justi\Documents\NEU\Fall_2024\MLFoundations\GroupProject\dataverse_files\MPIIFaceGaze\Data\Data'
# Make sure the dataset folder exists
if not os.path.exists(data_path):
    print("Dataset folder not found. Please place the dataset in 'data/dataset/'.")
dataset = MPIIFaceGazeDataset(root_dir=data_path, transform=transform)
# Split dataset into train and validation sets
train_size = int(0.7 * len(dataset)) # you an also modify size of the taining set
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

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

# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0.0
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#
#
#     # Validation
#     model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for data, target in val_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             val_loss += criterion(output, target).item()
#
#     print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')
#
# # Save the model
# torch.save(model.state_dict(), 'pigaze_model.pth')