import cvzone
import cv2
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot

cap = cv2.VideoCapture(0)

detector = FaceMeshDetector(maxFaces=1)
plotY = LivePlot(600, 500, [160, 272],invert=True)
idList = [22, 23, 24, 26, 110, 157, 158, 159, 160, 161, 130, 243]
ratioList = []
blink = 0
count = 0

while True:
    if cap.get(cv2.CAP_PROP_FRAME_COUNT) == cap.get(cv2.CAP_PROP_POS_FRAMES):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    success, img = cap.read()

    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        for id in idList:
            cv2.circle(img, face[id], 2, (0, 255, 0), cv2.FILLED)

        leftUp = face[159]
        leftDown = face[23]
        leftLeft = face[130]
        leftRight = face[243]
        lenghtHor, _ = detector.findDistance(leftUp, leftDown)
        lenghtVer, _ = detector.findDistance(leftLeft, leftRight)
        cv2.line(img, leftUp, leftDown, (255, 200, 0), 3)
        cv2.line(img, leftLeft, leftRight, (255, 200, 0), 3)

        ratio = int((lenghtVer / lenghtHor) * 100)
        ratioList.append(ratio)
        if len(ratioList) > 10:
            ratioList.pop(0)
        ratioAve = int(sum(ratioList) / len(ratioList))
        if ratioAve > 225 and count == 0:
            blink += 1
            count = 1
        if count != 0:
            count += 1
            if count == 20:
                count = 0

        cvzone.putTextRect(img, f"Blinks: {blink}", [20, 50], 2, 2, offset=20, border=2)

        imgPlot = plotY.update(ratioAve)
        # cv2.imshow("Plot", imgPlot)
        img = cv2.resize(img, (600, 500))
        imgStack = cvzone.stackImages([img, imgPlot], 2, 1)
    else:
        img = cv2.resize(img, (600, 500))
        imgStack = cvzone.stackImages([img, img], 2, 1)

    cv2.imshow("Image", imgStack)
    cv2.waitKey(1)
