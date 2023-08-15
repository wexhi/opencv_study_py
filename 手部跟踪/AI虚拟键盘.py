import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import random
import time
from pynput.keyboard import Controller

# def drawAll(img, buttonList):
#     for button in buttonList:
#         x, y = button.pos
#         w, h = button.size
#         cv2.rectangle(img, button.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
#         cv2.putText(img, button.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN,
#                     4, (255, 255, 255), 4)
#     return img

def drawAll(img, buttonList):
    imgNew = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        cvzone.cornerRect(imgNew, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                          20, rt=0)
        cv2.rectangle(imgNew, button.pos, (x + button.size[0], y + button.size[1]), (255, 0, 255), cv2.FILLED)
        cv2.putText(imgNew, button.text, (x + 40, y + 60), cv2.FONT_HERSHEY_PLAIN,
                    2, (255, 255, 255), 3)

    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    print(mask.shape)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]
    return out

class Button():
    def __init__(self, pos, text, size=[85, 85]):
        self.pos = pos
        self.size = size
        self.text = text

    def draw(self, img):
        x, y = self.pos
        w, h = self.size
        cv2.rectangle(img, self.pos, (x + w, y + h), (255, 0, 255), cv2.FILLED)
        cv2.putText(img, self.text, (self.pos[0] + 25, self.pos[1] + 60), cv2.FONT_HERSHEY_PLAIN,
                    4, (255, 255, 255), 4)
        return img



# Webcom
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)  # height

detector = HandDetector(detectionCon=0.8, maxHands=1)

keyList = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
           ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
           ["Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]

keyBoard = Controller()

buttonList = []
for i in range(len(keyList)):
    for j, key in enumerate(keyList[i]):
        buttonList.append(Button([j * 100 + 50, 100 * i + 50], key))

finalText = ""

delayCounter = 0

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip to see same side as us
    mylmList, img = detector.findHands(img, draw=True, flipType=False)
    if mylmList:
        lmList = mylmList[0]["lmList"]
        img = drawAll(img, buttonList)

        if lmList:
            for botton in buttonList:
                x,y = botton.pos
                w,h = botton.size

                if x < lmList[8][0] < x+w and y < lmList[8][1] < y+h:
                    cv2.rectangle(img, botton.pos, (x + w, y + h), (175, 0, 175), cv2.FILLED)
                    cv2.putText(img, botton.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN,
                                4, (255, 255, 255), 4)
                    l, _, _ = detector.findDistance(lmList[8][:2], lmList[12][:2], img)
                    print(l)

                    ## When clicked
                    if l < 30 and delayCounter == 0:
                        keyBoard.press(botton.text)
                        cv2.rectangle(img, botton.pos, (x + w, y + h), (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, botton.text, (x + 20, y + 65), cv2.FONT_HERSHEY_PLAIN,
                                    4, (255, 255, 255), 4)
                        finalText += botton.text
                        delayCounter = 1

    cv2.rectangle(img, (50, 350), (700, 450), (175, 0, 175), cv2.FILLED)
    cv2.putText(img, finalText, (60, 425), cv2.FONT_HERSHEY_PLAIN,
                5, (255, 255, 255), 5)
    # Avoid Duplicate Display
    if delayCounter > 0:
        delayCounter += 1
        if delayCounter > 15:
            delayCounter = 0

    cv2.imshow("Image", img)
    cv2.waitKey(1)
