import cv2
import mediapipe as mp
import time
import os
import HandTrackingModle as htm

folderPath = "images"
myList = os.listdir(folderPath)
# print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    # print(f'{folderPath}/{imPath}')
    # image = cv2.resize(image, (200, 200))
    overlayList.append(image)
print(len(overlayList))

cap = cv2.VideoCapture(0)

pTime = 0
cTime = 0

detector = htm.HandDetector(detectionCon=0.75)

tipIds = [4, 8, 12, 16, 20]  # 指尖的id

while True:
    success, img = cap.read() # 读取视频
    img = detector.findHands(img) # shape: 480, 640, 3
    lmList = detector.findPosition(img, draw=False)
    # print(lmList)
    if len(lmList) != 0:
        fingers = []
        # Thumb
        if lmList[tipIds[0]][1] > lmList[tipIds[0] - 1][1]:
            fingers.append(1)
            cv2.circle(img, (lmList[tipIds[0]][1], lmList[tipIds[0]][2]), \
                        15, (255, 0, 255), cv2.FILLED
                          )
        else:
            fingers.append(0)

        # 4 Fingers
        for id in range(1, 5):
            if lmList[tipIds[id]][2] < lmList[tipIds[id] - 2][2]:
                cv2.circle(img, (lmList[tipIds[id]][1], lmList[tipIds[id]][2]), \
                           15, (255, 0, 255), cv2.FILLED
                           )
                fingers.append(1)
            else:
                fingers.append(0)
        # print(fingers)
        totalFingers = fingers.count(1)
        print(totalFingers)

        h, w, c = overlayList[totalFingers].shape
        img[0:h, 0:w] = overlayList[totalFingers]

        cv2.rectangle(img, (490, 300), (640, 480), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(totalFingers), (515, 450),
                    cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)),
                (550, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                (255, 0, 255), 3)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.imshow("Image", img)
