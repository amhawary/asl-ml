import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import json

img_size = 200
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
padding = 50
data = {
    'A': [],
    'B': [],
    'C': [],
    'D': [],
    'E': [],
    'F': [],
    'G': [],
    'H': [],
    'I': [],
    'J': [],
    'K': [],
    'L': [],
    'M': [],
    'N': [],
    'O': [],
    'P': [],
    'Q': [],
    'R': [],
    'S': [],
    'T': [],
    'U': [],
    'V': [],
    'W': [],
    'X': [],
    'Y': [],
    'Z': [],
    'del': [],
    'space': [],
             }

while True:
    success, frame = cap.read()
    hands, frame = detector.findHands(frame)

    count = {
             'a': 0,
             'b': 0,
             'c': 0,
             'd': 0,
             'e': 0,
             'f': 0,
             'g': 0,
             'h': 0,
             'i': 0,
             'j': 0,
             'k': 0,
             'l': 0,
             'm': 0,
             'n': 0,
             'o': 0,
             'p': 0,
             'q': 0,
             'r': 0,
             's': 0,
             't': 0,
             'u': 0,
             'v': 0,
             'w': 0,
             'x': 0,
             'y': 0,
             'z': 0,
             'del':0,
             'space':0,
             }

    cv2.putText(frame, "a : "+str(count['a']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "b : "+str(count['b']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "c : "+str(count['c']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "d : "+str(count['d']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "e : "+str(count['e']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "f : "+str(count['f']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "g : "+str(count['g']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "h : "+str(count['h']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "i : "+str(count['i']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "k : "+str(count['k']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "l : "+str(count['l']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "m : "+str(count['m']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "n : "+str(count['n']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "o : "+str(count['o']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "p : "+str(count['p']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "q : "+str(count['q']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "r : "+str(count['r']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "s : "+str(count['s']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "t : "+str(count['t']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "u : "+str(count['u']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "v : "+str(count['v']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "w : "+str(count['w']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "x : "+str(count['x']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "y : "+str(count['y']), (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "z : "+str(count['z']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "del : "+str(count['del']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
    cv2.putText(frame, "space : "+str(count['space']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)

    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
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

    interrupt = cv2.waitKey(1)
    if interrupt & 0xFF == ord('a'):
        print("a")
        if hands:
            data["A"].append(hand["lmList"])
    if interrupt & 0xFF == ord('b'):
        print("b")
        if hands:
            data["B"].append(hand["lmList"])
    if interrupt & 0xFF == ord('c'):
        print("c")
        if hands:
            data["C"].append(hand["lmList"])
    if interrupt & 0xFF == ord('d'):
        print("d")
        if hands:
            data["D"].append(hand["lmList"])
    if interrupt & 0xFF == ord('e'):
        print("e")
        if hands:
            data["E"].append(hand["lmList"])
    if interrupt & 0xFF == ord('f'):
        print("f")
        if hands:
            data["F"].append(hand["lmList"])
    if interrupt & 0xFF == ord('g'):
        print("g")
        if hands:
            data["G"].append(hand["lmList"])
    if interrupt & 0xFF == ord('h'):
        print("h")
        if hands:
            data["H"].append(hand["lmList"])
    if interrupt & 0xFF == ord('i'):
        print("i")        
        if hands:
            data["I"].append(hand["lmList"])
    if interrupt & 0xFF == ord('j'):
        print("j")
        if hands:
            data["J"].append(hand["lmList"])
    if interrupt & 0xFF == ord('k'):
        print("k")
        if hands:
            data["K"].append(hand["lmList"])
    if interrupt & 0xFF == ord('l'):
        print("l")
        if hands:
            data["L"].append(hand["lmList"])
    if interrupt & 0xFF == ord('m'):
        print("m")
        if hands:
            data["M"].append(hand["lmList"])
    if interrupt & 0xFF == ord('n'):
        print("n")
        if hands:
            data["N"].append(hand["lmList"])
    if interrupt & 0xFF == ord('o'):
        print("o")
        if hands:
            data["O"].append(hand["lmList"])
    if interrupt & 0xFF == ord('p'):
        print("p")
        if hands:
            data["P"].append(hand["lmList"])
    if interrupt & 0xFF == ord('q'):
        print("q")
        if hands:
            data["Q"].append(hand["lmList"])
    if interrupt & 0xFF == ord('r'):
        print("r")
        if hands:
            data["R"].append(hand["lmList"])
    if interrupt & 0xFF == ord('s'):
        print("s")
        if hands:
            data["S"].append(hand["lmList"])
    if interrupt & 0xFF == ord('t'):
        print("t")
        if hands:
            data["T"].append(hand["lmList"])
    if interrupt & 0xFF == ord('u'):
        print("u")  
        if hands:
            data["U"].append(hand["lmList"])
    if interrupt & 0xFF == ord('v'):
        print("v")
        if hands:
            data["V"].append(hand["lmList"])
    if interrupt & 0xFF == ord('w'):
        print("w")
        if hands:
            data["W"].append(hand["lmList"])
    if interrupt & 0xFF == ord('x'):
        print("x")
        if hands:
            data["X"].append(hand["lmList"])
    if interrupt & 0xFF == ord('y'):
        print("y")
        if hands:
            data["Y"].append(hand["lmList"])
    if interrupt & 0xFF == ord('z'):
        print("z")
        if hands:
            data["Z"].append(hand["lmList"])
    if interrupt & 0xFF == 0x08:
        print("del")
        if hands:
            data["del"].append(hand["lmList"])
    if interrupt & 0xFF == 0x20:
        print("space")
        if hands:
            data["space"].append(hand["lmList"])
    if interrupt & 0xFF == 0x0D:
        print("Dump Data")
        with open('output.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

cap.release()
cv2.destroyAllWindows()