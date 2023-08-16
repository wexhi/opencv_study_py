import cv2
import cvzone
import numpy
from cvzone.FaceMeshModule import FaceMeshDetector
import numpy as np

cap = cv2.VideoCapture(0)

detector = FaceMeshDetector(maxFaces=1)

textList = ["Welcome to ", "Face Distance Measurement",
            "Here we will measure the distance", "between two points on your face",
            "Please keep your face straight", "and make sure your face is", "clearly visible"]

W = 6.3
d = 70
f = 380

sen = 10  # sensitivity

while True:
    success, img = cap.read()
    imgText = numpy.zeros_like(img)
    img, faces = detector.findFaceMesh(img, draw=False)

    if faces:
        face = faces[0]
        PointLeft = face[145]
        PointRight = face[374]
        # cv2.line(img, PointRight, PointLeft, (255, 0, 255), 2)
        # cv2.circle(img, PointLeft, 5, (0, 0, 255), cv2.FILLED)
        # cv2.circle(img, PointRight, 5, (0, 0, 255), cv2.FILLED)

        w, _ = detector.findDistance(PointLeft, PointRight)

        # Find distance between two points
        d = (W * f) / w
        print(d)
        cvzone.putTextRect(img, f"{int(d)} cm", [face[10][0] - 50, face[10][1] - 40],
                           2, 2, offset=20, border=2)

        for i, text in enumerate(textList):
            singleHeight = 30 + int((int(d / sen) * sen) / 5)
            scale = 0.4 + (int(d / sen) * sen) / 100
            cv2.putText(imgText, text, (20, 50 + (singleHeight * i)),
                        cv2.FONT_ITALIC, scale, (255, 255, 255), 2)

    imgStack = cvzone.stackImages([img, imgText], 2, 1)
    cv2.imshow("Image", imgStack)
    cv2.waitKey(1)
