import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from PIL import Image

img_size = 200
cap=cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
padding = 30

while True:
    # capture the current frame from the webcam
    ret, frame = cap.read()
    hands, frame = detector.findHands(frame)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        print(hand["lmList"][8])
        row = frame.shape[1]
        col = frame.shape[0]
        imgWhite = np.ones((img_size, img_size, 3), np.uint8) * 255
        imgCrop = frame[y - padding:y + h + padding, x - padding:x + w + padding]
        imgCropShape = imgCrop.shape
        if imgCrop.size > 0:
            imgCropResized = cv2.resize(imgCrop, (img_size, img_size))
            imgWhite[:, :] = imgCropResized
            cv2.imshow("ROI", imgWhite)

    cv2.imshow("data", frame)

    # Draw a rectangle around the ROI on the frame
    # cv2.rectangle(frame, rect_top_left, rect_bottom_right, (0, 255, 0), 2)
    # cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # cv2.imshow('ASL Recognition', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture device and close all windows
cap.release()
cv2.destroyAllWindows()

