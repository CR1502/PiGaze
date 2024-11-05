import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray

cam = PiCamera()
cam.resolution = (512,304)
cam.framerate = 10
rawCapture = PiRGBArray(cam, size=(512,304))



cap = cv2.VideoCapture(0)

while True:
	for frame in cam.capture_continuous(rawCapture, format="bgr", use_video_port=True):
		image = frame.array
		cv2.imshow("Press Space", image)
		rawCapture.truncate(0)

		k = cv2.waitKey(1)
		rawCapture.truncate(0)
		if (k%256 == 27):
			break
	if k%256 == 27:
		break
cv2.destroyAllWindows()


