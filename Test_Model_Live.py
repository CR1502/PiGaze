import cv2
import numpy as np
import torch
from MPIIH import *
from PIL import Image


trained_model = load_model('pigaze_model.pth')
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    #Extract image from frame
    image = cv2.imwrite('image.jpg', frame)

    #Make Prediction
    gaze_target = predict_gaze(trained_model, 'image.jpg', transform)

    #Print Model Predictions
    print(f"Gaze Target {gaze_target}")

    #Display Gaze Coordinates on frame
    x, y, = gaze_target[0]
    x = int(x)
    y = int(y)
    cv2.circle(frame, (x, y), 3, (255, 0, 0), 5)
    cv2.imshow("Frame", frame)

    #Press esc to kill script
    key = cv2.waitKey(1)

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()


