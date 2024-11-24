import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import torch
from PIL import Image
from MPII import *

cam = PiCamera()
cam.resolution = (512,304)
cam.framerate = 10
rawCapture = PiRGBArray(cam, size=(512,304))
trained_model = load_model('pigaze_model.pth')



cap = cv2.VideoCapture(0)

while True:
	for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		image1 = frame.array

		# Extract image from frame
		image = cv2.imwrite('image.jpg', image1)

		# Make Prediction
		gaze_target = predict_gaze(trained_model, 'image.jpg', transform)

		# Print Model Predictions
		print(f"Gaze Target {gaze_target}")

		# Display Gaze Coordinates on frame
		x, y, = gaze_target[0]
		x = int(x)
		y = int(y)
		cv2.circle(frame, (x, y), 3, (255, 0, 0), 5)
		cv2.imshow("Frame", frame)
		cv2.imshow("Press Space", image)
		rawCapture.truncate(0)

		k = cv2.waitKey(1)
		rawCapture.truncate(0)
		if (k%256 == 27):
			break
	if k%256 == 27:
		break
cv2.destroyAllWindows()

