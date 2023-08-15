import cv2
import numpy as np
import time
import os
import HandTrackingModle as htm

brushThickness = 15
eraseThickness = 50

folderPath = "images"
Header = cv2.imread("images/canva.jpg")
drawColor = (255, 0, 255)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

detector = htm.HandDetector(detectionCon=0.75)
xp, yp = 0, 0
imgCanvas = np.zeros((480, 640, 3), np.uint8)

while True:
    # 1. Import image
    success, img = cap.read()  # 读取视频
    img = cv2.flip(img, 1)

    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        # tip of index and middle fingers
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)

        # 4. If Selection Mode - Two finger are up
        if fingers[1] and fingers[2]:
            xp, yp = 0, 0
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25),
                          drawColor, cv2.FILLED)
            print("Selection Mode")
            if y1 < 100:
                if x1 < 180:
                    cv2.putText(img, "RED", (30, 110),
                                cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 255), 3)
                    drawColor = (0, 0, 255)
                    print("RED")
                elif 180 <= x1 < 430:
                    cv2.putText(img, "BLUE", (200, 110),
                                cv2.FONT_HERSHEY_PLAIN, 3,
                                (255, 0, 0), 3)
                    drawColor = (255, 0, 0)
                    print("BLUE")
                elif 430 <= x1 < 640:
                    cv2.putText(img, "ERASE", (400, 110),
                                cv2.FONT_HERSHEY_PLAIN, 3,
                                (0, 0, 0), 3)
                    drawColor = (0, 0, 0)
                    print("ERASE")
            cv2.putText(img, "Selection Mode", (30, 130),
                        cv2.FONT_HERSHEY_PLAIN, 3,
                        drawColor, 3)


        # 5. If Drawing Mode - Index finger up
        if fingers[1] and fingers[2] == False:
            cv2.circle(img, (x1, y1), 15, drawColor, cv2.FILLED)
            print("Drawing Mode")

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            if drawColor == (0, 0, 0):
                cv2.line(img, (x1, y1), (xp, yp), drawColor, eraseThickness)
                cv2.line(imgCanvas, (x1, y1), (xp, yp), drawColor, eraseThickness)
            else:
                cv2.line(img, (x1, y1), (xp, yp), drawColor, brushThickness)
                cv2.line(imgCanvas, (x1, y1), (xp, yp), drawColor, brushThickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    img[0:100, 0:640] = Header
    # img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0) # 透明度

    cv2.imshow("Image", img)
    # cv2.imshow("Canvas", imgCanvas)
    cv2.waitKey(1)
