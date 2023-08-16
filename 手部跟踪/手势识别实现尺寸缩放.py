import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.8)
startDistance = None
scale = 0
img1 = cv2.imread("images/1.png")
cx, cy = 500, 500

while True:
    success, img = cap.read()
    # img = cv2.flip(img, 1)
    hands, img = detector.findHands(img, flipType=True)

    if len(hands) == 2:
        # print(detector.fingersUp(hands[0]), detector.fingersUp(hands[1]))
        if detector.fingersUp(hands[0]) == [1, 1, 1, 1, 1] and \
                detector.fingersUp(hands[1]) == [1, 1, 1, 1, 1]:
            # print("Zooming")
            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]
            # point1 = lmList1[8] is index fingertip
            if startDistance is None:
                length, info, img = detector.findDistance(lmList1[8][:2], lmList2[8][:2], img)

                startDistance = length

            length, info, img = detector.findDistance(lmList1[8][:2], lmList2[8][:2], img)
            scale = int((length - startDistance) // 2)
            cx, cy = info[4:]
            print(scale)
            startDistance = length
    else:
        startDistance = None

    try:
        h1, w1, _ = img1.shape
        newH, newW = ((h1 + scale) // 2) * 2, ((w1 + scale) // 2) * 2
        img1 = cv2.resize(img1, (newW, newH))
        img[cy - newH // 2:cy + newH // 2, cx - newW // 2:cx + newW // 2] = img1
        scale = 0
    except:
        pass

    cv2.imshow("Image", img)
    cv2.waitKey(1)
