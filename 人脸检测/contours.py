import cv2
import numpy as np


def getContours(img, cThr=[100, 100], showCanny=True, minArea=1000, filter=0, draw=False):
    img = cv2.resize(img, (0, 0), None, 0.4, 0.4)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, cThr[0], cThr[1])
    kernel = np.ones((7, 7))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=4)
    img_threshold = cv2.erode(img_dilate, kernel, iterations=3)

    contours, hiearchy = cv2.findContours(img_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    finalContours = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > minArea:
            peri = cv2.arcLength(i, True)
            appox = cv2.approxPolyDP(i, 0.02 * peri, True)
            bbox = cv2.boundingRect(appox)
            if filter > 0:
                if len(appox) == filter:
                    finalContours.append([len(appox), area, appox, bbox, i])
            else:
                finalContours.append([len(appox), area, appox, bbox, i])
    finalContours = sorted(finalContours, key=lambda x: x[1], reverse=True)
    if draw:
        for con in finalContours:
            cv2.drawContours(img, con[4], -1, (0, 0, 255), 3)
    if showCanny: cv2.imshow("Canny", img_canny)
    return img, finalContours


def reorder(myPoints):
    print(myPoints.shape)
    myPoints_new = np.zeros_like(myPoints)
    myPoints = myPoints.reshape((4,2))
    add = myPoints.sum(1)
    myPoints_new[0] = myPoints[np.argmin(add)]
    myPoints_new[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    myPoints_new[1] = myPoints[np.argmin(diff)]
    myPoints_new[2] = myPoints[np.argmax(diff)]
    return myPoints_new


def warpImg(img, points, w, h):
    print(points)
    reorder(points)
