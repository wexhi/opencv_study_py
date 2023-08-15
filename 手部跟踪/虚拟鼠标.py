import cv2
import numpy as np
import time
import HandTrackingModle as htm
import pyautogui

wCam, hCam = 640, 480
frameR = 100  # Frame Reduction
smoothening = 4

plocX, plocY = 0, 0
clocX, clocY = 0, 0

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
cTime = 0

detector = htm.HandDetector(maxHands=1)

wScreen, hScreen = pyautogui.size()
print(wScreen, hScreen)

while True:
    # 1. Import image,Find Hand Landmarks
    sussess, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmList, bbox = detector.findPosition(img, draw=False)

    # 2.Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        # print(x1,y1,x2,y2)

        # 3.Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR), (wCam - frameR, hCam - frameR),
                      (255, 0, 255), 2)
        # 4.Only Index Finger : Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5.Convert Coordinates
            x3 = np.interp(x1, (frameR, wCam - frameR), (0, wScreen))
            y3 = np.interp(y1, (frameR, hCam - frameR), (0, hScreen))

            # 6.Smooth Trajectory
            clocX = plocX + (x3 - plocX) / smoothening
            clocY = plocY + (y3 - plocY) / smoothening

            # 7.Move Mouse
            pyautogui.moveTo( clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)

            plocX, plocY = clocX, clocY

        # 8.five fingers : Clicking Mode
        if fingers[0] and fingers[1] and fingers[2] and fingers[3] and fingers[4]:
            # 9.Find distance between fingers
            lenth, img, _ = detector.findDistance(8, 12, img, draw=True)
            print(lenth)
            # 10.Click mouse if distance short
            if lenth < 40:
                cv2.circle(img, (x1, y1), 15, (0, 255, 0), cv2.FILLED)
                pyautogui.click()

    # 11.Frame rate
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),
                (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 0), 3)
    # 12.Display
    cv2.imshow("Image", img)
    cv2.waitKey(1)
