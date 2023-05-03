import os
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np

img_size = 200
cap = cv2.VideoCapture(0)
directory = 'Dataset 3/'
detector = HandDetector(maxHands=1)
padding = 50

while True:
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    hands, frame = detector.findHands(frame)

    count = {
             'a': len(os.listdir(directory+"/A")),
             'b': len(os.listdir(directory+"/B")),
             'c': len(os.listdir(directory+"/C")),
             'd': len(os.listdir(directory+"/D")),
             'e': len(os.listdir(directory+"/E")),
             'f': len(os.listdir(directory+"/F")),
             'g': len(os.listdir(directory+"/G")),
             'h': len(os.listdir(directory+"/H")),
             'i': len(os.listdir(directory+"/I")),
             'j': len(os.listdir(directory+"/J")),
             'k': len(os.listdir(directory+"/K")),
             'l': len(os.listdir(directory+"/L")),
             'm': len(os.listdir(directory+"/M")),
             'n': len(os.listdir(directory+"/N")),
             'o': len(os.listdir(directory+"/O")),
             'p': len(os.listdir(directory+"/P")),
             'q': len(os.listdir(directory+"/Q")),
             'r': len(os.listdir(directory+"/R")),
             's': len(os.listdir(directory+"/S")),
             't': len(os.listdir(directory+"/T")),
             'u': len(os.listdir(directory+"/U")),
             'v': len(os.listdir(directory+"/V")),
             'w': len(os.listdir(directory+"/W")),
             'x': len(os.listdir(directory+"/X")),
             'y': len(os.listdir(directory+"/Y")),
             'z': len(os.listdir(directory+"/Z"))
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
        cv2.imwrite(directory+'A/'+str(count['a'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('b'):
        print("b")
        cv2.imwrite(directory+'B/'+str(count['b'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('c'):
        print("c")
        cv2.imwrite(directory+'C/'+str(count['c'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('d'):
        print("d")
        cv2.imwrite(directory+'D/'+str(count['d'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('e'):
        print("e")
        cv2.imwrite(directory+'E/'+str(count['e'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('f'):
        print("f")
        cv2.imwrite(directory+'F/'+str(count['f'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('g'):
        print("g")
        cv2.imwrite(directory+'G/'+str(count['g'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('h'):
        print("h")
        cv2.imwrite(directory+'H/'+str(count['h'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('i'):
        print("i")        
        cv2.imwrite(directory+'I/'+str(count['i'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('j'):
        print("j")
        cv2.imwrite(directory+'J/'+str(count['j'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('k'):
        print("k")
        cv2.imwrite(directory+'K/'+str(count['k'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('l'):
        print("l")
        cv2.imwrite(directory+'L/'+str(count['l'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('m'):
        print("m")
        cv2.imwrite(directory+'M/'+str(count['m'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('n'):
        print("n")
        cv2.imwrite(directory+'N/'+str(count['n'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('o'):
        print("o")
        cv2.imwrite(directory+'O/'+str(count['o'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('p'):
        print("p")
        cv2.imwrite(directory+'P/'+str(count['p'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('q'):
        print("q")
        cv2.imwrite(directory+'Q/'+str(count['q'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('r'):
        print("r")
        cv2.imwrite(directory+'R/'+str(count['r'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('s'):
        print("s")
        cv2.imwrite(directory+'S/'+str(count['s'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('t'):
        print("t")
        cv2.imwrite(directory+'T/'+str(count['t'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('u'):
        print("u")  
        cv2.imwrite(directory+'U/'+str(count['u'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('v'):
        print("v")
        cv2.imwrite(directory+'V/'+str(count['v'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('w'):
        print("w")
        cv2.imwrite(directory+'W/'+str(count['w'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('x'):
        print("x")
        cv2.imwrite(directory+'X/'+str(count['x'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('y'):
        print("y")
        cv2.imwrite(directory+'Y/'+str(count['y'])+'.png',imgWhite)
    if interrupt & 0xFF == ord('z'):
        print("z")
        cv2.imwrite(directory+'Z/'+str(count['z'])+'.png',imgWhite)

cap.release()
cv2.destroyAllWindows()