import cv2
import os
from cvzone.HandTrackingModule import HandDetector
import numpy as np

# Variables
width = 1280
height = 720
folderPath = "presentation"

# Camera settings
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Get the list of presentation images
pathImages = sorted(os.listdir(folderPath), key=len)
print(pathImages)

# Variables
imgNumber = 0
hs, ws = 120, 213
gestureThreshold = 300  # Threshold for gesture control
buttonPressed = False
buttonCounter = 0
buttonDelay = 15
annotation = [[]]
annotationNumber = -1
annotationStarted = False

# Hand detector
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
    imgCurrent = cv2.imread(pathFullImage)

    # Find Hand
    hands, img = detector.findHands(img, flipType=False)
    cv2.line(img, (0, gestureThreshold), (width, gestureThreshold),
             (255, 0, 0), 2)
    if hands and not buttonPressed:
        hand = hands[0]
        fingers = detector.fingersUp(hand)
        cx, cy = hand["center"]
        lmList = hand["lmList"]

        # Constraint the index finger easier to control
        indexFinger = lmList[8][:2]
        xVal = int(np.interp(indexFinger[0], [width // 2, width - 200], [0, width]))
        yVal = int(np.interp(indexFinger[1], [100, height - 300], [0, height]))
        indexFinger = xVal, yVal
        print(fingers)

        if cy <= gestureThreshold:  # if hand is above gesture threshold

            # Gesture 1 - Left
            if fingers == [0, 0, 0, 0, 0]:
                print("Left")
                if imgNumber > 0:
                    annotation = [[]]
                    annotationNumber = -1
                    annotationStarted = False
                    imgNumber -= 1
                    buttonPressed = True
            # Gesture 2 - Right
            elif fingers == [1, 0, 0, 0, 1]:
                print("Right")
                if imgNumber < len(pathImages) - 1:
                    annotation = [[]]
                    annotationNumber = -1
                    annotationStarted = False
                    imgNumber += 1
                    buttonPressed = True
        # Gesture 3 - Show Pointer
        if fingers == [1, 1, 1, 0, 0] or fingers == [0, 1, 1, 0, 0]:
            cv2.circle(imgCurrent, indexFinger,
                       12, (177, 0, 177), cv2.FILLED)
            annotationStarted = False
        # Gesture 4 - Draw
        if fingers == [1, 1, 0, 0, 0]:
            if annotationStarted is False:
                annotationStarted = True
                annotationNumber += 1
                annotation.append([])
            cv2.circle(imgCurrent, indexFinger,
                       16, (255, 0, 255), cv2.FILLED)
            annotation[annotationNumber].append(indexFinger)
        else:
            annotationStarted = False
        # Gesture 5 - Erase
        if fingers == [0, 1, 1, 1, 1]:
            if annotation:
                annotation.pop()
                annotationNumber -= 1
                buttonPressed = True
                if annotationNumber < -1:
                    annotationNumber = -1

    # Button released
    if buttonPressed:
        buttonCounter += 1
        if buttonCounter >= buttonDelay:
            buttonPressed = False
            buttonCounter = 0

    for i in range(len(annotation)):
        for j in range(len(annotation[i])):
            if j > 0:
                cv2.line(imgCurrent, annotation[i][j], annotation[i][j - 1], (0, 0, 255), 12)

    # Adding webcam image to the presentation image
    imgSmall = cv2.resize(img, (ws, hs))
    h, w, _ = imgCurrent.shape
    imgCurrent[0:hs, w - ws:] = imgSmall

    cv2.imshow("Image", img)
    cv2.imshow("Slides", imgCurrent)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
