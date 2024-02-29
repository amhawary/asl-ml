import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
from PIL import Image
import json
import csv

img_size = 200
cap=cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
padding = 30

f = open('data.csv', 'w')
writer = csv.writer(f)

class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'space']
data = {}
writer.writerow("class, 0x, 0y, 0z, 1x, 1y, 1z, 2x, 2y, 2z, 3x, 3y, 3z, 4x, 4y, 4z, 5x, 5y, 5z, 6x, 6y, 6z, 7x, 7y, 7z, 8x, 8y, 8z, 9x, 9y, 9z, 10x, 10y, 10z, 11x, 11y, 11z, 12x, 12y, 12z, 13x, 13y, 13z, 14x, 14y, 14z, 15x, 15y, 15z, 16x, 16y, 16z, 17x, 17y, 17z, 18x, 18y, 18z, 19x, 19y, 19z, 20x, 20y, 20z, 21x, 21y, 21z")
for letter in class_names:
        directory1 = str("asl-ml/Datasets/Original Dataset/asl_alphabet_train/asl_alphabet_train/"+letter+"/")
        count = 0
        data[letter] = []
        for img in os.listdir(directory1):
            image = Image.open(str(directory1+img))
            image.load()
            frame = np.asarray(image)
            hands, frame = detector.findHands(frame)
            if hands:
                hand = hands[0]
                if hand:
                    line = letter + ", "
                    data[letter].append(hand["lmList"])
                    for pnt in hand["lmList"]:
                         for i in pnt:
                            line = line + str(i) + ", "
                    writer.writerow(line)
                    count+=1
                    print(letter,"-",count)
with open('data.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)