import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import math
import numpy as np
import random
import time

# Webcom
cap = cv2.VideoCapture(0)
cap.set(3, 1280)  # width
cap.set(4, 720)  # height

# Hand Tracking
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Find Function
# x is the raw distance y is the value in cm
x = [200, 126, 104, 87, 72, 55, 47, 35]
y = [14, 21, 29, 36, 44, 53, 68, 72]
coff = np.polyfit(x, y, 2)

# Game Variables
cx, cy = 250, 250
color = (255, 0, 255)
counter = 0
score = 0
timeStart = time.time()
total_time = 20

# Loop
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip to see same side as us

    if time.time() - timeStart < total_time:
        hands = detector.findHands(img, draw=False)  # with draw

        if hands:
            lmList = hands[0]["lmList"]  # Left hand landmarks
            x, y, w, h = hands[0]["bbox"]
            # print(lmList)
            x1, y1 = lmList[5][:2]
            x2, y2 = lmList[17][:2]
            distance = int(math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))
            A, B, C = coff
            distanceCM = A * distance ** 2 + B * distance + C

            # print(distance,distanceCM)

            if distanceCM < 35:
                if x < cx < x + w and y < cy < y + h:
                    counter = 1

            cvzone.putTextRect(img, f"{int(distanceCM)} cm", (x + 5, y - 10))
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 3)

        if counter:
            counter += 1
            color = (0, 255, 0)
            if counter == 3:
                cx = random.randint(100, 1100)
                cy = random.randint(100, 600)
                score += 1
                color = (255, 0, 255)
                counter = 0

        # Draw Buttons
        cv2.circle(img, (cx, cy), 30, color, cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 255, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 20, (255, 255, 255), 2)
        cv2.circle(img, (cx, cy), 30, (50, 55, 55), 2)

        # Game HUD
        cvzone.putTextRect(img, f"Time: {int(total_time - time.time() + timeStart)}", \
                           (1000, 75), scale=3, offset=20)
        cvzone.putTextRect(img, f"Score: {str(score).zfill(2)}", (60, 75), scale=3, offset=20)
    else:
        cvzone.putTextRect(img, "Game Over", (400, 400), scale=5, offset=30, thickness=7)
        cvzone.putTextRect(img, f"Your Score: {score}", (350, 500), scale=5, offset=20)
        cvzone.putTextRect(img, "Press 'R' to Restart", (460, 575), scale=2, offset=10)
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)

    if key == 27:
        break
    if key == ord('r'):
        score = 0
        timeStart = time.time()
