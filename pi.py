import cv2
import numpy as np
#import picamera
from picamera2 import Picamera2
from picamera2 import Preview
#from picamera.array import PiRGBArray
import subprocess
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def load_model(model_path):
    model = PiGazeModel()
    model.load_state_dict(torch.load(model_path, map_location = torch.device('cpu')))
    model.eval()
    return model

def predict_gaze(model, image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
    gaze_target = output[:, :2].numpy()
    return gaze_target

#Add whatever model we end up with here
class PiGazeModel(nn.Module):
    def __init__(self):
        super(PiGazeModel, self).__init__()
        #layout for pigaze_model5(final)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.SiLU(inplace=True),
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

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


picam2 = Picamera2()
config = picam2.create_preview_configuration({'size': (640,480)} )
picam2.configure(config)
picam2.start()
trained_model = load_model('pigaze_model.pth')


while True:
	while True:
		frame = picam2.capture_array()

		frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# Extract image from frame
		image = cv2.imwrite('image.jpg', frame_rgb)

		# Make Prediction
		gaze_target = predict_gaze(trained_model, 'image.jpg', transform)

		# Print Model Predictions
		print(f"Gaze Target {gaze_target}")

		# Display Gaze Coordinates on frame
		x, y, = gaze_target[0]
		x = int(x)
		y = int(y)
		cv2.circle(frame_rgb, (x, y), 3, (255, 0, 0), 5)
		cv2.imshow("Frame", frame_rgb)

		k = cv2.waitKey(1)
		if (k%256 == 27):
			break

	break
picam2.stop()
#out.release()
cv2.destroyAllWindows()

