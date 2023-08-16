import cv2
from cvzone.HandTrackingModule import HandDetector
import cvzone
import numpy as np

class DragRect():
    def __init__(self, posCenter, size=[200, 200], colorR=(255, 0, 255)):
        self.posCenter = posCenter
        self.size = size
        self.colorR = colorR

    def update(self, cursor):
        cx, cy = self.posCenter
        w, h = self.size

        # If the index finger tip is in the rectangle region
        if cx - w // 2 < cursor[0] < cx + w // 2 and \
                cy - h // 2 < cursor[1] < cy + h // 2:
            self.colorR = (0, 255, 0)
            # If the index finger tip is in the upper half of the rectangle region
            if detector.fingersUp(hands[0]) == [0, 1, 1, 1, 1] or \
                    detector.fingersUp(hands[0]) == [1, 1, 1, 1, 1]:
                self.posCenter = cursor
        else:
            self.colorR = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8, maxHands=1)

colorR = (255, 0, 255)
cx, cy, w, h = 100, 100, 200, 200

DragRectList = []
for x in range(5):
    DragRectList.append(DragRect([x * 250 + 150, 150]))

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, draw=True, flipType=False)
    if hands:
        print(detector.fingersUp(hands[0]))
        limList = hands[0]['lmList']
        cursor = limList[8][:2]
        for rect in DragRectList:
            rect.update(cursor)

    # Draw solid rectangle
    # for rect in DragRectList:
    #     cx, cy = rect.posCenter
    #     w, h = rect.size
    #     colorR = rect.colorR
    #     cv2.rectangle(img, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2),
    #                   colorR, cv2.FILLED)
    #     cvzone.cornerRect(img, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)

    # Draw transparent rectangle
    imgNew = np.zeros_like(img, np.uint8)
    for rect in DragRectList:
        cx, cy = rect.posCenter
        w, h = rect.size
        colorR = rect.colorR
        cv2.rectangle(imgNew, (cx - w // 2, cy - h // 2), (cx + w // 2, cy + h // 2),
                      colorR, cv2.FILLED)
        cvzone.cornerRect(imgNew, (cx - w // 2, cy - h // 2, w, h), 20, rt=0)
    out = img.copy()
    alpha = 0.5
    mask = imgNew.astype(bool)
    out[mask] = cv2.addWeighted(img, alpha, imgNew, 1 - alpha, 0)[mask]

    # cv2.imshow('frame', img)
    cv2.imshow('frame', out)
    if cv2.waitKey(1) == ord('q'):
        break
